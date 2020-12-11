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
    def __init__(self, mid_frame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mid_frame = mid_frame

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
        x_2d = x[:, :, self.mid_frame, :, :]  # Main frame for detection
        x_3d = x  # Input clip

        if self.training:
            x_2d = checkpoint(forward_backbone, self.backbone, x_2d)
            x_3d = checkpoint(self.backbone_3d, x_3d)
        else:
            x_2d = self.backbone(x_2d)
            x_3d = self.backbone_3d(x_3d)

        feature_maps = []
        for i, (x_2d_res, x_3d_res) in enumerate(zip(x_2d, x_3d)):
            x_3d_res = x_3d_res[:, :, 0, :, :]
            x = torch.cat((x_3d_res, x_2d_res), dim=1)
            x = self.cfams[i](x) + x_2d_res
            feature_maps += [x]

        # EffDet components
        x = self.fpn(feature_maps)
        x_class = self.class_net(x)
        x_box = self.box_net(x)
        return x_class, x_box
