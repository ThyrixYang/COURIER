import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
import torch.distributed as dist

from models import get_model
import torchmetrics.functional as mF

from pl_module_base import RecPLModuleBase


class RecPLModule(RecPLModuleBase):

    def __init__(self,
                 args):
        super().__init__(args)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.cfg.train.lr, 
            weight_decay=self.cfg.train.weight_decay
        )
        return optimizer

    def cal_emb(self, image):
        emb = self.image_backbone_forward(image)
        return emb

    def training_step(self, batch, batch_idx):
        batch = self.backbone_forward(batch)
        user_batch_index = batch["user_batch_index"]
        pv_batch_index = batch["pv_batch_index"]
        click_list = batch["click_list"]

        user_seq_emb = batch["user_seq_emb"]
        pv_seq_emb = batch["pv_seq_emb"]

        # user query joint embedding
        pred_loss_dict = self.pred_loss_fn(
            user_seq_emb=user_seq_emb,
            pv_seq_emb=pv_seq_emb,
            pv_batch_index=pv_batch_index,
            click_list=click_list,
            user_seq_emb_mask=batch["user_seq_emb_mask"],
            pv_seq_emb_mask=batch["pv_seq_emb_mask"]
        )

        contrastive_loss_dict = self.contrastive_loss(
            user_seq_emb=user_seq_emb,
            pv_seq_emb=pv_seq_emb,
            pv_batch_index=pv_batch_index,
            click_list=click_list,
            user_seq_emb_mask=batch["user_seq_emb_mask"],
            pv_seq_emb_mask=batch["pv_seq_emb_mask"]
        )

        user_seq_loss_dict = self.user_seq_loss(user_seq_emb=user_seq_emb)

        all_loss = pred_loss_dict["loss"] + contrastive_loss_dict["loss"] + user_seq_loss_dict["loss"]
        self.log("train/all_loss", 
            all_loss, 
            logger=True, 
            sync_dist=True, batch_size=1)

        for n, p in self.named_parameters():
            self.log(f"model/{n}_mean", p.mean())
            self.log(f"model/{n}_std", p.std())

        return all_loss

    def user_seq_loss(self, user_seq_emb):
        if "user_seq_loss_weight" not in self.cfg.model or self.cfg.model.user_seq_loss_weight < 1e-8:
            return {"loss": 0}
        user_seq_emb = user_seq_emb.view(
            (-1, self.cfg.data.max_user_seq_len, self.cfg.model.emb_size))
        user_seq_target_emb = user_seq_emb[:, 0]
        user_seq_input_emb = user_seq_emb[:, 1:]
        reconstruct_pv_emb = self.attn_layer(
            query=user_seq_target_emb.unsqueeze(1),
            key=user_seq_input_emb,
            value=user_seq_input_emb,
        )[0].mean(1)

        pv_seq_emb = F.normalize(user_seq_target_emb, p=2, dim=1)
        reconstruct_pv_emb = F.normalize(reconstruct_pv_emb, p=2, dim=1)

        #### dist gather #############################################
        if self.cfg.model.dist_batch:
            tensor_dict = {
                "user_emb": reconstruct_pv_emb,
                "pv_seq_emb": pv_seq_emb,
            }
            gathered_dict = self.gather_all_tensors_with_grad(
                tensor_dict, cat_res=True)
            reconstruct_pv_emb = gathered_dict["user_emb"]
            pv_seq_emb = gathered_dict["pv_seq_emb"]
        ###############################################################

        M = pv_seq_emb @ reconstruct_pv_emb.t()

        label_M = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
        pos_num = torch.sum(label_M)

        tau = self.cfg.model.tau
        self.logger.log_histograms(
            tag="train/user_seq_logits_dist",
            value=M,
            step=self.global_step
        )
        self.logger.log_histograms(
            tag="train/user_seq_logits_dist_tau",
            value=M*tau,
            step=self.global_step
        )

        log_prob = F.log_softmax(M/tau, dim=0)
        pos_probs = label_M * torch.diag(log_prob.exp())

        loss = torch.sum((-log_prob * label_M)) / pos_num * self.cfg.model.user_seq_loss_weight
        self.logger.log_metrics(
            {
                "user_seq_loss/pos_prob": pos_probs.sum() / pos_num,
                "user_seq_loss/contrastive_loss": loss,
                "user_seq_loss/user_seq_proj_size": user_seq_emb.shape[0],
                "user_seq_loss/pv_seq_proj_size": pv_seq_emb.shape[0],
            },
            step=self.global_step
        )
        return {
            "loss": loss
        }

    def contrastive_loss(self,
                      user_seq_emb,
                      pv_seq_emb,
                      click_list,
                      pv_batch_index,
                      user_seq_emb_mask,
                      pv_seq_emb_mask):
        if self.cfg.model.contrastive_weight < 1e-8:
            return {"loss": 0}

        user_seq_emb = user_seq_emb.view(
            (-1, self.cfg.data.max_user_seq_len, self.cfg.model.emb_size))
        reconstruct_pv_emb = self.attn_layer(
            query=pv_seq_emb.unsqueeze(1),
            key=user_seq_emb[pv_batch_index],
            value=user_seq_emb[pv_batch_index],
        )[0].mean(1)

        pv_seq_emb = F.normalize(pv_seq_emb, p=2, dim=1)
        reconstruct_pv_emb = F.normalize(reconstruct_pv_emb, p=2, dim=1)

        #### dist gather #############################################
        if self.cfg.model.dist_batch:
            tensor_dict = {
                "user_emb": reconstruct_pv_emb,
                "pv_seq_emb": pv_seq_emb,
                "click_list": click_list,
                "pv_seq_emb_mask": pv_seq_emb_mask
            }
            gathered_dict = self.gather_all_tensors_with_grad(
                tensor_dict, cat_res=True)
            reconstruct_pv_emb = gathered_dict["user_emb"]
            pv_seq_emb = gathered_dict["pv_seq_emb"]
            click_list = gathered_dict["click_list"]
            pv_seq_emb_mask = gathered_dict["pv_seq_emb_mask"]
        ###############################################################

        M = pv_seq_emb @ reconstruct_pv_emb.t()

        masked_label = click_list.float() * pv_seq_emb_mask.float()
        label_M = torch.diag(masked_label)
        pos_num = torch.sum(label_M)

        tau = self.cfg.model.tau
        self.logger.log_histograms(
            tag="train/logits_dist",
            value=M,
            step=self.global_step
        )
        self.logger.log_histograms(
            tag="train/logits_dist_tau",
            value=M/tau,
            step=self.global_step
        )

        log_prob = F.log_softmax(M*tau, dim=0)
        pos_probs = masked_label * torch.diag(log_prob.exp())

        loss = torch.sum((-log_prob * label_M)) / pos_num * self.cfg.model.contrastive_weight
        self.logger.log_metrics(
            {
                "contrastive/pos_prob": pos_probs.sum() / pos_num,
                "contrastive/contrastive_loss": loss,
                "contrastive/user_seq_proj_size": user_seq_emb.shape[0],
                "contrastive/pv_seq_proj_size": pv_seq_emb.shape[0],
            },
            step=self.global_step
        )
        return {
            "loss": loss
        }