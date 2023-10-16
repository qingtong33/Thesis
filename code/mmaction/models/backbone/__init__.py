# Copyright (c) OpenMMLab. All rights reserved.
from .agcn import AGCN
from .c3d import C3D
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v2_tsm import MobileNetV2TSM
from .resnet import ResNet
from .resnet2plus1d import ResNet2Plus1d
from .resnet3d import ResNet3d, ResNet3dLayer
from .resnet3d_csn import ResNet3dCSN
from .resnet3d_slowfast import ResNet3dSlowFast
from .resnet3d_slowonly import ResNet3dSlowOnly
from .resnet_audio import ResNetAudio
from .resnet_tin import ResNetTIN
from .resnet_tsm import ResNetTSM
from .stgcn import STGCN
from .tanet import TANet
from .timesformer import TimeSformer
from .x3d import X3D
from .replknet import RepLKNet
from .replknet_tsm_v1 import RepLKNetTSMV1
from .replknet_tsm_v2 import RepLKNetTSMV2
from .replknet_tsm_v3 import RepLKNetTSMV3
from .replknet_tsm_v1R import RepLKNetTSMV1R
from .replknet_tsm_v2R import RepLKNetTSMV2R
from .replknet_tsm_v3R import RepLKNetTSMV3R

__all__ = [
    'C3D', 'ResNet', 'ResNet3d', 'ResNetTSM', 'ResNet2Plus1d',
    'ResNet3dSlowFast', 'ResNet3dSlowOnly', 'ResNet3dCSN', 'ResNetTIN', 'X3D',
    'ResNetAudio', 'ResNet3dLayer', 'MobileNetV2TSM', 'MobileNetV2', 'TANet',
    'TimeSformer', 'STGCN', 'AGCN', 'RepLKNet', 'RepLKNetTSMV1', 'RepLKNetTSMV2'
    , 'RepLKNetTSMV3', 'RepLKNetTSMV1R', 'RepLKNetTSMV2R', 'RepLKNetTSMV3R'
]
