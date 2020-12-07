import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.config import set_config_readonly, set_config_writeable

from .yowo import YOWO

def get_net(model_cfg, num_classes):
    config = get_efficientdet_config(model_cfg.efficientdet_config)
    net = YOWO(config, pretrained_backbone=True)

    if model_cfg.pretrained_2d != '':
        load_state_dict(net, model_cfg.pretrained)

    set_config_writeable(config)
    config.num_classes = num_classes
    config.image_size = model_cfg.img_size
    net.reset_head(num_classes=num_classes)
    set_config_readonly(config)

    if model_cfg.start_from_2d != '':
        load_state_dict(net, model_cfg.start_from)

    if model_cfg.freeze_first_epoch:
        freeze(net.backbone)
        freeze(net.fpn)

    return DetBenchTrain(net, config)


def load_state_dict(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']

    # model.load_state_dict(checkpoint)
    soft_load_state_dict(model, checkpoint)


def soft_load_state_dict(model, state_dict):
    model_state = model.state_dict()

    for name, param in state_dict.items():
        if name not in model_state or model_state[name].shape != param.shape:
            continue

        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        model_state[name].copy_(param)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True