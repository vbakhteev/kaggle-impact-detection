import torch
import torch.nn as nn
from effdet import EfficientDet
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from src.model import resnext
from src.model.cfam import CFAMBlock

def forward_backbone(backbone, x):
    out = backbone(x)
    return tuple(out)


class YOWO(EfficientDet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ##### 2D Backbone #####
        out = self.backbone(torch.zeros(1, 3, 256, 256))
        num_ch_2d = (o.shape[1] for o in out)  # Number of output channels for backbone_2d

        ##### 3D Backbone #####
        self.backbone_3d = resnext.resnext101()
        num_ch_3d = (512, 1024, 2048)  # Number of output channels for backbone_3d

        ##### Attention & Final Conv #####
        self.cfams = nn.ModuleList()
        for ch_2d, ch_3d in zip(num_ch_2d, num_ch_3d):
            cfam = CFAMBlock(ch_2d + ch_3d, ch_2d)
            self.cfams.append(cfam)

    def forward(self, x):
        mid_i = (x.size(2) - 1) // 2
        x_2d = x[:, :, mid_i, :, :]  # Middle frame of the clip
        x_3d = x  # Input clip

        x_2d = checkpoint(forward_backbone, self.backbone, x_2d)
        x_3d = checkpoint(self.backbone_3d, x_3d)

        feature_maps = []
        for i, (x_2d_res, x_3d_res) in enumerate(zip(x_2d, x_3d)):
            x_3d_res = x_3d_res[:, :, 0, :, :]
            x = torch.cat((x_3d_res, x_2d_res), dim=1)
            x = self.cfams[i](x)
            feature_maps += [x]

        # EffDet components
        x = self.fpn(feature_maps)
        x_class = self.class_net(x)
        x_box = self.box_net(x)
        return x_class, x_box


def get_fine_tuning_parameters(model, opt):
    ft_module_names = ['cfam', 'conv_final']  # Always fine tune 'cfam' and 'conv_final'
    if not opt.freeze_backbone_2d:
        ft_module_names.append('backbone_2d')  # Fine tune complete backbone_3d
    else:
        ft_module_names.append('backbone_2d.models.29')  # Fine tune only layer 29 and 30
        ft_module_names.append('backbone_2d.models.30')  # Fine tune only layer 29 and 30

    if not opt.freeze_backbone_3d:
        ft_module_names.append('backbone_3d')  # Fine tune complete backbone_3d
    else:
        ft_module_names.append('backbone_3d.layer4')  # Fine tune only layer 4

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters
