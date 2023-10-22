#num_segments in this file should equal to config
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import _BatchNorm
import torch.utils.checkpoint as checkpoint #二选一 建议改成cp跟随mmaction
#from torch.utils import checkpoint as cp
from timm.models.layers import DropPath#模型库文件
import sys
import os
from ...utils import get_root_logger
from ..builder import BACKBONES
#from mmaction.registry import MODELS

# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# Based on ConvNeXt, timm, DINO and DeiT code bases
# https://github.com/facebookresearch/ConvNeXt
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        #   Please follow the instructions https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/README.md
        #   export LARGE_KERNEL_CONV_IMPL=absolute_path_to_where_you_cloned_the_example (i.e., depthwise_conv2d_implicit_gemm.py)
        # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull requests are welcomed.
        # Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

use_sync_bn = False

def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = False

def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

def shift(x, num_segments=8, shift_div=8):
    """Perform temporal shift operation on the feature.

    Args:
        x (torch.Tensor): The input feature to be shifted.
        num_segments (int): Number of frame segments.
        shift_div (int): Number of divisions for shift. Default: 3.

    Returns:
        torch.Tensor: The shifted feature.
    """
    # [N, C, H, W]
    #print("xxxx")
    n, c, h, w = x.size()

    # [N // num_segments, num_segments, C, H*W]
    # can't use 5 dimensional array on PPL2D backend for caffe
    x = x.view(-1, num_segments, c, h * w)

    # get shift fold
    fold = c // shift_div

    # split c channel into three parts:
    # left_split, mid_split, right_split
    left_split = x[:, :, :fold, :]
    mid_split = x[:, :, fold:2 * fold, :]
    right_split = x[:, :, 2 * fold:, :]

    # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
    # because array on caffe inference must be got by computing

    # shift left on num_segments channel in `left_split`
    zeros = left_split - left_split
    blank = zeros[:, :1, :, :]
    left_split = left_split[:, 1:, :, :]
    left_split = torch.cat((left_split, blank), 1)

    # shift right on num_segments channel in `mid_split`
    zeros = mid_split - mid_split
    blank = zeros[:, :1, :, :]
    mid_split = mid_split[:, :-1, :, :]
    mid_split = torch.cat((blank, mid_split), 1)

    # right_split: no shift

    # concatenate
    out = torch.cat((left_split, mid_split, right_split), 2)

    # [N, C, H, W]
    # restore the original dimension
    return out.view(n, c, h, w)



class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2, groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class ConvFFN(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels, drop_path, num_segments=8, shift_div=8):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = get_bn(in_channels)
        self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.nonlinear = nn.GELU()
        self.num_segments=num_segments
        self.shift_div=shift_div

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKBlock(nn.Module):

    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path, small_kernel_merged=False, num_segments=8, shift_div=8):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1, groups=dw_channels, small_kernel=small_kernel, small_kernel_merged=small_kernel_merged,)
        self.lk_nonlinear = nn.ReLU()
        self.prelkb_bn = get_bn(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.num_segments=num_segments
        self.shift_div=shift_div
        print('drop path:', self.drop_path)

    def forward(self, x):
        out = shift(x, self.num_segments, self.shift_div)
        out = self.prelkb_bn(out)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKNetStage(nn.Module):

    def __init__(self, channels, num_blocks, stage_lk_size, drop_path,
                 small_kernel, dw_ratio=1, ffn_ratio=4,
                 use_checkpoint=False,      # train with torch.utils.checkpoint to save memory
                 small_kernel_merged=False,
                 norm_intermediate_features=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        blks = []
        for i in range(num_blocks):
            block_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            #   Assume all RepLK Blocks within a stage share the same lk_size. You may tune it on your own model.
            replk_block = RepLKBlock(in_channels=channels, dw_channels=int(channels * dw_ratio), block_lk_size=stage_lk_size,
                                     small_kernel=small_kernel, drop_path=block_drop_path, small_kernel_merged=small_kernel_merged)
            convffn_block = ConvFFN(in_channels=channels, internal_channels=int(channels * ffn_ratio), out_channels=channels,
                                    drop_path=block_drop_path)
            blks.append(replk_block)
            blks.append(convffn_block)
        self.blocks = nn.ModuleList(blks)
        if norm_intermediate_features:
            self.norm = get_bn(channels)    #   Only use this with RepLKNet-XL on downstream tasks
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)   # Save training memory
            else:
                x = blk(x)
        return x


@BACKBONES.register_module()
class RepLKNetTSMV2R(nn.Module):

    def __init__(self, large_kernel_sizes, layers, channels, drop_path_rate, small_kernel,
                 dw_ratio=1, ffn_ratio=4, in_channels=3, num_classes=1000, out_indices=None,
                 use_checkpoint=False,
                 pretrained=None,
                 small_kernel_merged=False,
                 use_sync_bn=True,
                 norm_intermediate_features=False,     # for RepLKNet-XL on COCO and ADE20K, use an extra BN to normalize the intermediate feature maps then feed them into the heads
                 torchvision_pretrain = False,
                 frozen_stages=-1,# frozen_stages (int): Stages to be frozen (all param fixed). If set to -1, it means not freezing any parameters. Default: -1.
                 #layers=[2,2,18,2], 大概是有4个stage
                 num_segments=8,
                 is_shift=True,
                 shift_div=8,
                 ):
        super().__init__()

        if num_classes is None and out_indices is None:
            raise ValueError('must specify one of num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and out_indices is not None:
            raise ValueError('cannot specify both num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and norm_intermediate_features:
            raise ValueError('for pretraining, no need to normalize the intermediate feature maps')
        self.out_indices = out_indices
        if use_sync_bn:
            enable_sync_bn()

        base_width = channels[0]
        self.frozen_stages = frozen_stages
        self.torchvision_pretrain = torchvision_pretrain
        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint
        self.norm_intermediate_features = norm_intermediate_features
        self.num_stages = len(layers)
        self.stem = nn.ModuleList([
            conv_bn_relu(in_channels=in_channels, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=1, padding=1, groups=base_width),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=1, stride=1, padding=0, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=base_width)])
        # stochastic depth. We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        #shift 参数
        self.num_segments = num_segments
        self.is_shift = is_shift
        self.shift_div = shift_div

        for stage_idx in range(self.num_stages):
            layer = RepLKNetStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
                                  stage_lk_size=large_kernel_sizes[stage_idx],
                                  drop_path=dpr[sum(layers[:stage_idx]):sum(layers[:stage_idx + 1])],
                                  small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                                  use_checkpoint=use_checkpoint, small_kernel_merged=small_kernel_merged,
                                  norm_intermediate_features=norm_intermediate_features)
            self.stages.append(layer)
            if stage_idx < len(layers) - 1:
                transition = nn.Sequential(
                    conv_bn_relu(channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1),
                    conv_bn_relu(channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1, groups=channels[stage_idx + 1]))
                self.transitions.append(transition)

        #self.init_structure()
        """去掉head初始化
        if num_classes is not None:
            self.norm = get_bn(channels[-1])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(channels[-1], num_classes)
        """

#以下为初始化参数方法
    def _load_torchvision_checkpoint(self, logger=None):
        """Initiate the parameters from torchvision pretrained checkpoint."""
        state_dict_torchvision = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_torchvision:
            state_dict_torchvision = state_dict_torchvision['state_dict']

        """loaded_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if 'downsample' in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + '.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace('conv', 'bn')
                self._load_conv_params(module.conv, state_dict_torchvision,
                                       original_conv_name, loaded_param_names)
                self._load_bn_params(module.bn, state_dict_torchvision,
                                     original_bn_name, loaded_param_names)
        """
        #这部分涉及名称转换的部分，不弄

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(
            state_dict_torchvision.keys()) - set(loaded_param_names)
        if remaining_names:
            logger.info(
                f'These parameters in pretrained checkpoint are not loaded'
                f': {remaining_names}')

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            if self.torchvision_pretrain:
                # torchvision's
                self._load_torchvision_checkpoint(logger)
            else:
                # ours
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


#以上为初始化参数方法

    

    def forward_features(self, x):
        x = self.stem[0](x)
        for stem_layer in self.stem[1:]:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(stem_layer, x)     # save memory
            else:
                x = stem_layer(x)

        if self.out_indices is None:
            #   Just need the final output
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return x
        else:
            #   Need the intermediate feature maps
            outs = []
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx in self.out_indices:
                    outs.append(self.stages[stage_idx].norm(x))     # For RepLKNet-XL normalize the features before feeding them into the heads
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return outs

    def forward(self, x):
        x = self.forward_features(x)
        """if self.out_indices:
            return x
        else:
            x = self.norm(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.head(x) backbone#文件去掉分类头
            return x
        """
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    #   If your framework cannot automatically fuse BN for inference, you may do it manually.
    #   The BNs after and before conv layers can be removed.
    #   No need to call this if your framework support automatic BN fusion.
    def deep_fuse_BN(self):
        for m in self.modules():
            if not isinstance(m, nn.Sequential):
                continue
            if not len(m) in [2, 3]:  # Only handle conv-BN or conv-BN-relu
                continue
            #   If you use a custom Conv2d impl, assume it also has 'kernel_size' and 'weight'
            if hasattr(m[0], 'kernel_size') and hasattr(m[0], 'weight') and isinstance(m[1], nn.BatchNorm2d):
                conv = m[0]
                bn = m[1]
                fused_kernel, fused_bias = fuse_bn(conv, bn)
                fused_conv = get_conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                                        stride=conv.stride,
                                        padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True)
                fused_conv.weight.data = fused_kernel
                fused_conv.bias.data = fused_bias
                m[0] = fused_conv
                m[1] = nn.Identity()
