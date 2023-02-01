import os
import torch
import argparse
from pprint import pprint

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy

from pl_module_ctr import RecPLModule

def main(args):
    torch.set_printoptions(threshold=10000)
    cfg = get_config(args.config)

    ##################################################
    # config progress bar
    callbacks = []
    callbacks.append(TQDMProgressBar(refresh_rate=cfg.logging.log_every_n_steps))


    model = RecPLModule(args)
    print("train from fresh")
    
    datamodule = DataModule(args)
    ###################################################

    trainer = Trainer(
        accelerator="gpu",
        devices=args.gpu_per_node,  # x gpus per node
        num_nodes=args.worker_count,  # x machine
        max_epochs=cfg.train.max_epochs,
        max_steps=-1,  # disable
        sync_batchnorm=True,
        precision=cfg.train.precision,
        fast_dev_run=False,
        
        # logging
        log_every_n_steps=cfg.logging.log_every_n_steps,
        enable_progress_bar=True,
        callbacks=callbacks,
        enable_checkpointing=False,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain")
    parser.add_argument("config", type=str, help="py config file")

    parser.add_argument("--worker_count", type=str)
    parser.add_argument("--gpu_per_node", type=int)

    args = parser.parse_args()
    main(args)
