import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.config import set_config_readonly, set_config_writeable
from torch.utils.checkpoint import checkpoint

from .yowo import YOWO


def get_net(model_cfg, num_classes, mid_frame):
    config = get_efficientdet_config(model_cfg.efficientdet_config)
    net = YOWO(mid_frame, config, pretrained_backbone=False)

    if model_cfg.pretrained_effdet != '':
        load_state_dict(net, model_cfg.pretrained_effdet)
    if model_cfg.pretrained_backbone_3d != '':
        load_state_dict(net, model_cfg.pretrained_backbone_3d)

    set_config_writeable(config)
    config.num_classes = num_classes
    config.image_size = model_cfg.img_size
    net.reset_head(num_classes=num_classes)
    set_config_readonly(config)

    if model_cfg.start_from != '':
        load_state_dict(net, model_cfg.start_from)

    if model_cfg.freeze_backbone_2d:
        freeze(net.backbone)
    if model_cfg.freeze_backbone_3d:
        freeze(net.backbone_3d)

    return DetBenchTrain(net, config)


class EfficientDetCheckpoined(EfficientDet):
    def forward_backbone(self, x):
        x = self.backbone(x)
        return tuple(x)

    def forward(self, x):
        x = checkpoint(self.forward_backbone, x)
        x = self.fpn(list(x))
        x_class = self.class_net(x)
        x_box = self.box_net(x)
        return x_class, x_box


def get_effdet(model_cfg, num_classes):
    config = get_efficientdet_config(model_cfg.efficientdet_config)
    net = EfficientDetCheckpoined(config, pretrained_backbone=True)

    if model_cfg.pretrained_effdet != '':
        load_state_dict(net, model_cfg.pretrained_effdet)

    set_config_writeable(config)
    config.num_classes = num_classes
    config.image_size = model_cfg.img_size
    net.reset_head(num_classes=num_classes)
    set_config_readonly(config)

    if model_cfg.start_from != '':
        load_state_dict(net, model_cfg.start_from)

    if model_cfg.freeze_backbone_2d:
        freeze(net.backbone)

    return DetBenchTrain(net, config)


def load_state_dict(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    # model.load_state_dict(checkpoint)
    soft_load_state_dict(model, checkpoint)


def soft_load_state_dict(model, state_dict):
    model_state = model.state_dict()

    not_loaded_params = []
    for name, param in state_dict.items():
        if name.startswith('module.'):
            name = name[7:]

        if name not in model_state or model_state[name].shape != param.shape:
            not_loaded_params += [name]
            continue

        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        model_state[name].copy_(param)

    if len(not_loaded_params):
        print("WARNING: following params couldn't loaded into model:", not_loaded_params)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
