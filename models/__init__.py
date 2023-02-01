import torchvision.models as models
import torch.nn as nn

def get_model(name, cfg):
    if name == "swin_t":
        from models import swin_t, Swin_T_Weights
        weights = Swin_T_Weights.DEFAULT if cfg.model.use_image_pretrain else None
        model = swin_t(weights=weights)
        return model
    else:
        raise NotImplementedError()