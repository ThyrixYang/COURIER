import gc
import math
from datetime import date, datetime, timedelta
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torch.distributed as dist
import pytorch_lightning.utilities.distributed as pl_dist
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import pprint

from models import get_model
from models.ctrnet import CTRNet
import torchmetrics.functional as mF


class RecPLModuleBase(LightningModule):

    def __init__(self,
                 args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.cfg = get_config(args.config)

        self.log_values = {}

        # essential models
        self.init_models()

    def forward(self, batch):
        raise NotImplementedError()

    def gather_index(self, index):
        gather_index = pl_dist.gather_all_tensors(index)
        for i in range(len(gather_index) - 1):
            gather_index[i+1] += torch.max(gather_index[i]) + 1
        return torch.cat(gather_index)

    def gather_all_tensors_with_grad(self, tensor_dict, cat_res=True):
        group = torch.distributed.group.WORLD

        tensor_dict = {k: v.contiguous() for k, v in tensor_dict.items()}

        world_size = torch.distributed.get_world_size(group)
        torch.distributed.barrier(group=group)

        # 1. Gather sizes of all tensors
        local_size_dict = {k: torch.tensor(
            v.shape, device=v.device) for k, v in tensor_dict.items()}
        local_sizes_dict = {
            k: [torch.zeros_like(v) for _ in range(world_size)]
            for k, v in local_size_dict.items()
        }
        torch.distributed.barrier(group=group)
        for k in local_size_dict.keys():
            torch.distributed.all_gather(local_sizes_dict[k],
                                         local_size_dict[k],
                                         group=group,
                                         async_op=True)
        torch.distributed.barrier(group=group)
        max_size_dict = {
            k: torch.stack(local_sizes).max(dim=0).values
            for k, local_sizes in local_sizes_dict.items()
        }

        # 2. We need to pad each local tensor to maximum size, gather and then truncate
        tensor_dict_padded = {}
        for k in tensor_dict.keys():
            pad_dims = []
            pad_by = (max_size_dict[k] - local_size_dict[k]).detach().cpu()
            for val in reversed(pad_by):
                pad_dims.append(0)
                pad_dims.append(val.item())
            tensor_dict_padded[k] = F.pad(tensor_dict[k], pad_dims)
        torch.distributed.barrier(group=group)
        gathered_result_dict = self.all_gather(
            tensor_dict_padded, group=group, sync_grads=True)
        res_dict = {
            k: [None for _ in range(v.shape[0])]
            for k, v in gathered_result_dict.items()
        }
        for k in tensor_dict.keys():
            for idx, item_size in enumerate(local_sizes_dict[k]):
                slice_param = [slice(dim_size) for dim_size in item_size]
                res_dict[k][idx] = gathered_result_dict[k][idx][slice_param]
            if cat_res:
                res_dict[k] = torch.cat(res_dict[k], dim=0)
        return res_dict

    def _gather_all_tensors_with_grad(self, result):
        """
        pad, gather, trim, and concatenate
        """
        group = torch.distributed.group.WORLD

        # convert tensors to contiguous format
        result = result.contiguous()

        world_size = torch.distributed.get_world_size(group)
        torch.distributed.barrier(group=group)

        # 1. Gather sizes of all tensors
        local_size = torch.tensor(result.shape, device=result.device)
        local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(local_sizes, local_size, group=group)
        max_size = torch.stack(local_sizes).max(dim=0).values

        # 2. We need to pad each local tensor to maximum size, gather and then truncate
        pad_dims = []
        pad_by = (max_size - local_size).detach().cpu()
        for val in reversed(pad_by):
            pad_dims.append(0)
            pad_dims.append(val.item())
        result_padded = F.pad(result, pad_dims)
        gathered_result = self.all_gather(
            result_padded, group=group, sync_grads=True)
        res = [None for _ in range(gathered_result.shape[0])]
        for idx, item_size in enumerate(local_sizes):
            slice_param = [slice(dim_size) for dim_size in item_size]
            res[idx] = gathered_result[idx][slice_param]
        res = torch.cat(res, dim=0)
        return res

    def load_backbone(self):
        self.image_backbone = get_model(
            self.cfg.model.image_backbone,
            self.cfg
        )
        self.image_backbone_head = nn.Linear(768, self.cfg.model.emb_size)

    def init_models(self):
        self.load_backbone()
        self.attn_layer = nn.MultiheadAttention(self.cfg.model.emb_size,
                                               num_heads=4,
                                               batch_first=True)
        self.ctr = CTRNet(in_dim=self.cfg.model.emb_size*2, out_dim=1)

    def image_backbone_forward(self, x):
        if self.cfg.model.fix_image_backbone:
            self.image_backbone.eval()
            with torch.no_grad():
                emb = self.image_backbone(x)
        else:
            emb = self.image_backbone(x)
        emb = self.image_backbone_head(emb)
        return emb

    def backbone_forward(self, batch):
        res = {}
        pv_seq = batch["pv_seq"]
        user_seq = batch["user_seq"]

        flat_pv_seq = pv_seq.view((-1, 3*224*224))
        flat_user_seq = user_seq.view((-1, 5, 3*224*224))
        res["user_seq_emb_mask"] = torch.logical_not(
            torch.any(flat_user_seq.bool(), dim=2))
        res["pv_seq_emb_mask"] = torch.any(flat_pv_seq.bool(), dim=1)
        label_n_mask = batch["click_list"] < 0
        click_list = batch["click_list"]
        click_list[label_n_mask] = 0
        res["click_list"] = click_list
        res["pv_seq_emb_mask"] = torch.logical_and(torch.logical_not(label_n_mask), 
                res["pv_seq_emb_mask"])

        pv_seq_len = pv_seq.shape[0]

        all_image = torch.cat([pv_seq, user_seq], dim=0)
        all_image_emb = self.image_backbone_forward(all_image)
        res["user_seq_emb"] = all_image_emb[pv_seq_len:]
        res["pv_seq_emb"] = all_image_emb[:pv_seq_len]

        batch.update(res)
        return batch

    def pred_loss_fn(self,
                     user_seq_emb,
                     pv_seq_emb,
                     click_list,
                     pv_batch_index,
                     user_seq_emb_mask,
                     pv_seq_emb_mask):
        user_seq_emb = user_seq_emb.view(
            (-1, self.cfg.data.max_user_seq_len, self.cfg.model.emb_size))

        reconstruct_pv_emb = self.attn_layer(
            query=pv_seq_emb.unsqueeze(1),
            key=user_seq_emb[pv_batch_index],
            value=user_seq_emb[pv_batch_index],
        )[0].mean(1)

        net_input = torch.cat(
            [reconstruct_pv_emb, pv_seq_emb], dim=1)

        logits = self.ctr(net_input)
        loss = F.binary_cross_entropy_with_logits(
            logits, click_list.view((-1, 1)).type_as(logits), reduction="none")
        loss = torch.sum(loss.view(-1) * pv_seq_emb_mask.float()) \
             / torch.sum(pv_seq_emb_mask.float()) \
             * self.cfg.model.ctr_weight
        prob = torch.sigmoid(logits)

        _prob = prob[pv_seq_emb_mask]
        _click_list = click_list[pv_seq_emb_mask]

        gather_probs = torch.cat(pl_dist.gather_all_tensors(_prob), dim=0)
        gather_labels = torch.cat(
            pl_dist.gather_all_tensors(_click_list), dim=0)
        auc = mF.auroc(preds=gather_probs.view((-1)),
                       target=gather_labels.view((-1)), 
                       num_classes=2)

        self.logger.log_metrics(
            {
                "pred/avg_prob": gather_probs.mean(),
                "pred/loss": loss,
                "pred/auc": auc,
                "pred/avg_labels": gather_labels.float().mean(),
                "pred/pv_emb_mask_mean": pv_seq_emb_mask.float().mean(),
            },
            step=self.global_step
        )

        self.log("pred_loss", loss, on_step=True, prog_bar=True, on_epoch=True)
        self.log("pred_auc", auc, on_step=True, prog_bar=True, on_epoch=True)
        return {
            "loss": loss
        }

    def configure_optimizers(self):
        opt_name = self.cfg.train.optimizer
        if opt_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.train.lr,
                weight_decay=self.cfg.train.weight_decay
            )
        else:
            raise ValueError()
        return optimizer

    def pad_tensors(self, tensor_dict, batch_size):
        tensor_dict["pv_pad_mask"] = torch.cat(
            [torch.ones(tensor_dict["click_list"].shape[0]),
             torch.zeros(self.cfg.data.max_pv_seq_len*batch_size-tensor_dict["click_list"].shape[0])]).to(self.device)
        tensor_dict["user_pad_mask"] = torch.cat(
            [torch.ones(tensor_dict["search_click_seq"].shape[0]),
             torch.zeros(self.cfg.data.max_user_seq_len*batch_size-tensor_dict["search_click_seq"].shape[0])]).to(self.device)
        for k in tensor_dict.keys():
            if k in ["click_list", "pv_seq_emb_proj", "pv_batch_index"]:
                max_size = self.cfg.data.max_pv_seq_len*batch_size
            elif k in ["search_click_seq_emb_proj", "search_click_batch_index", "user_query_joint_emb_proj"]:
                max_size = self.cfg.data.max_user_seq_len*batch_size
            else:
                continue
            t = tensor_dict[k]
            tensor_dict[k] = torch.cat(
                [t,
                 torch.zeros((max_size-t.shape[0], *t.shape[1:])).type_as(t)])
        return tensor_dict

    def cat_gathered_tensors(self, tensor_dict, batch_size):
        tensor_dict["pv_batch_index"] = (tensor_dict["pv_batch_index"] +
                                         torch.arange(dist.get_world_size(), device=self.device).view((-1, 1)).type_as(
            tensor_dict["pv_batch_index"]) * batch_size).view((-1, 1))
        tensor_dict["search_click_batch_index"] = (tensor_dict["search_click_batch_index"] +
                                                   torch.arange(dist.get_world_size(), device=self.device).view((-1, 1)).type_as(
            tensor_dict["search_click_batch_index"]) * batch_size).view((-1, 1))
        for k in tensor_dict.keys():
            if k in ["pv_batch_index", "search_click_batch_index"]:
                continue
            else:
                tensor_dict[k] = tensor_dict[k].view(
                    (-1, *tensor_dict[k].shape[2:]))
        return tensor_dict
