"""
Backbone modules.
"""
from collections import OrderedDict
from typing import Any, Optional, Tuple, Callable
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from einops import rearrange
from fvcore.nn.squeeze_excitation import SqueezeExcitation
import math
from torchvision.ops import RoIAlign
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from util.misc import NestedTensor
# class NestedTensor(object):
#     def __init__(self, tensors, mask: Optional[Tensor]):
#         self.tensors = tensors
#         self.mask = mask

#     def to(self, device):
#         # type: (Device) -> NestedTensor # noqa
#         cast_tensor = self.tensors.to(device)
#         mask = self.mask
#         if mask is not None:
#             assert mask is not None
#             cast_mask = mask.to(device)
#         else:
#             cast_mask = None
#         return NestedTensor(cast_tensor, cast_mask)

#     def decompose(self):
#         return self.tensors, self.mask

#     def __repr__(self):
#         return str(self.tensors)

from .position_encoding import build_position_encoding

def set_attributes(self, params: List[object] = None) -> None:
    """
    An utility function used in classes to set attributes from the input list of parameters.
    Args:
        params (list): list of parameters.
    """
    if params:
        for k, v in params.items():
            if k != "self":
                setattr(self, k, v)

def round_width(width, multiplier, min_width=8, divisor=8, ceil=False):
    """
    Round width of filters based on width multiplier
    Args:
        width (int): the channel dimensions of the input.
        multiplier (float): the multiplication factor.
        min_width (int): the minimum width after multiplication.
        divisor (int): the new width should be dividable by divisor.
        ceil (bool): If True, use ceiling as the rounding method.
    """
    if not multiplier:
        return width

    width *= multiplier
    min_width = min_width or divisor
    if ceil:
        width_out = max(min_width, int(math.ceil(width / divisor)) * divisor)
    else:
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)

def round_repeats(repeats, multiplier):
    """
    Round number of layers based on depth multiplier.
    """
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

class Conv2plus1d(nn.Module):
    """
    Implementation of 2+1d Convolution by factorizing 3D Convolution into an 1D temporal
    Convolution and a 2D spatial Convolution with Normalization and Activation module
    in between:

    ::

                        Conv_t (or Conv_xy if conv_xy_first = True)
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                        Conv_xy (or Conv_t if conv_xy_first = True)

    The 2+1d Convolution is used to build the R(2+1)D network.
    """

    def __init__(
        self,
        *,
        conv_t: nn.Module = None,
        norm: nn.Module = None,
        activation: nn.Module = None,
        conv_xy: nn.Module = None,
        conv_xy_first: bool = False,
    ) -> None:
        """
        Args:
            conv_t (torch.nn.modules): temporal convolution module.
            norm (torch.nn.modules): normalization module.
            activation (torch.nn.modules): activation module.
            conv_xy (torch.nn.modules): spatial convolution module.
            conv_xy_first (bool): If True, spatial convolution comes before temporal conv
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.conv_t is not None
        assert self.conv_xy is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_xy(x) if self.conv_xy_first else self.conv_t(x)
        x = self.norm(x) if self.norm else x
        x = self.activation(x) if self.activation else x
        x = self.conv_t(x) if self.conv_xy_first else self.conv_xy(x)
        return x

class Swish(nn.Module):
    """
    Wrapper for the Swish activation function.
    """

    def forward(self, x):
        return SwishFunction.apply(x)

class SwishFunction(torch.autograd.Function):
    """
    Implementation of the Swish activation function: x * sigmoid(x).

    Searching for activation functions. Ramachandran, Prajit and Zoph, Barret
    and Le, Quoc V. 2017
    """

    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))

class ResNetBasicHead(nn.Module):
    """
    ResNet basic head. This layer performs an optional pooling operation followed by an
    optional dropout, a fully-connected projection, an optional activation layer and a
    global spatiotemporal averaging.

    ::

                                        Pool3d
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
                                           ↓
                                       Averaging

    The builder can be found in `create_res_basic_head`.
    """

    def __init__(
        self,
        pool: nn.Module = None,
        dropout: nn.Module = None,
        proj: nn.Module = None,
        activation: nn.Module = None,
        output_pool: nn.Module = None,
    ) -> None:
        """
        Args:
            pool (torch.nn.modules): pooling module.
            dropout(torch.nn.modules): dropout module.
            proj (torch.nn.modules): project module.
            activation (torch.nn.modules): activation module.
            output_pool (torch.nn.Module): pooling module for output.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.proj is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Performs pooling.
        if self.pool is not None:
            x = self.pool(x)
        # Performs dropout.
        if self.dropout is not None:
            x = self.dropout(x)
        # Performs projection.
        if self.proj is not None:
            x = x.permute((0, 2, 3, 4, 1))
            x = self.proj(x)
            x = x.permute((0, 4, 1, 2, 3))
        # Performs activation.
        if self.activation is not None:
            x = self.activation(x)

        if self.output_pool is not None:
            # Performs global averaging.
            x = self.output_pool(x)
            x = x.view(x.shape[0], -1)
        return x

class Net(nn.Module):
    """
    Build a general Net models with a list of blocks for video recognition.

    ::

                                         Input
                                           ↓
                                         Block 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Block N
                                           ↓

    The ResNet builder can be found in `create_resnet`.
    """

    def __init__(self, *, blocks: nn.ModuleList) -> None:
        """
        Args:
            blocks (torch.nn.module_list): the list of block modules.
        """
        super().__init__()
        assert blocks is not None
        self.blocks = blocks
        init_net_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = {}
        for _, block in enumerate(self.blocks):
            x = block(x)
            out[str(_)] = x
        return out

class SpatioTemporalClsPositionalEncoding(nn.Module):
    """
    Add a cls token and apply a spatiotemporal encoding to a tensor.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_embed_shape: Tuple[int, int, int],
        sep_pos_embed: bool = False,
        has_cls: bool = True,
    ) -> None:
        """
        Args:
            embed_dim (int): Embedding dimension for input sequence.
            patch_embed_shape (Tuple): The number of patches in each dimension
                (T, H, W) after patch embedding.
            sep_pos_embed (bool): If set to true, one positional encoding is used for
                spatial patches and another positional encoding is used for temporal
                sequence. Otherwise, only one positional encoding is used for all the
                patches.
            has_cls (bool): If set to true, a cls token is added in the beginning of each
                input sequence.
        """
        super().__init__()
        assert (
            len(patch_embed_shape) == 3
        ), "Patch_embed_shape should be in the form of (T, H, W)."
        self.cls_embed_on = has_cls
        self.sep_pos_embed = sep_pos_embed
        self._patch_embed_shape = tuple(patch_embed_shape)
        self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
        self.num_temporal_patch = patch_embed_shape[0]

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_patches = self.num_spatial_patch * self.num_temporal_patch + 1
        else:
            self.cls_token = torch.tensor(0)
            num_patches = self.num_spatial_patch * self.num_temporal_patch

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.num_spatial_patch, embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.num_temporal_patch, embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
            else:
                self.pos_embed_class = torch.tensor([])  # for torchscriptability
            self.pos_embed = torch.tensor([])

        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            # Placeholders for torchscriptability, won't be used
            self.pos_embed_spatial = torch.tensor([])
            self.pos_embed_temporal = torch.tensor([])
            self.pos_embed_class = torch.tensor([])

    @torch.jit.export
    def patch_embed_shape(self) -> Tuple[int, int, int]:
        return self._patch_embed_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        B, N, C = x.shape
        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.num_temporal_patch, 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.num_spatial_patch,
                dim=1,
            )
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        return x

def _init_resnet_weights(model: nn.Module, fc_init_std: float = 0.01) -> None:
    """
    Performs ResNet style weight initialization. That is, recursively initialize the
    given model in the following way for each type:
        Conv - Follow the initialization of kaiming_normal:
            https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_
        BatchNorm - Set weight and bias of last BatchNorm at every residual bottleneck
            to 0.
        Linear - Set weight to 0 mean Gaussian with std deviation fc_init_std and bias
            to 0.
    Args:
        model (nn.Module): Model to be initialized.
        fc_init_std (float): the expected standard deviation for fully-connected layer.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            c2_msra_fill(m)
        elif isinstance(m, nn.modules.batchnorm._NormBase):
            if m.weight is not None:
                if hasattr(m, "block_final_bn") and m.block_final_bn:
                    m.weight.data.fill_(0.0)
                else:
                    m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            if hasattr(m, "xavier_init") and m.xavier_init:
                c2_xavier_fill(m)
            else:
                m.weight.data.normal_(mean=0.0, std=fc_init_std)
            if m.bias is not None:
                m.bias.data.zero_()
    return model

def _init_vit_weights(model: nn.Module, trunc_normal_std: float = 0.02) -> None:
    """
    Weight initialization for vision transformers.

    Args:
        model (nn.Module): Model to be initialized.
        trunc_normal_std (float): the expected standard deviation for fully-connected
            layer and ClsPositionalEncoding.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=trunc_normal_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, SpatioTemporalClsPositionalEncoding):
            for weights in m.parameters():
                nn.init.trunc_normal_(weights, std=trunc_normal_std)

def init_net_weights(
    model: nn.Module,
    init_std: float = 0.01,
    style: str = "resnet",
) -> None:
    """
    Performs weight initialization. Options include ResNet style weight initialization
    and transformer style weight initialization.

    Args:
        model (nn.Module): Model to be initialized.
        init_std (float): The expected standard deviation for initialization.
        style (str): Options include "resnet" and "vit".
    """
    assert style in ["resnet", "vit"]
    if style == "resnet":
        return _init_resnet_weights(model, init_std)
    elif style == "vit":
        return _init_vit_weights(model, init_std)
    else:
        raise NotImplementedError

class ResBlock(nn.Module):
    """
    Residual block. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.

    ::


                                         Input
                                           |-------+
                                           ↓       |
                                         Block     |
                                           ↓       |
                                       Summation ←-+
                                           ↓
                                       Activation

    The builder can be found in `create_res_block`.
    """

    def __init__(
        self,
        branch1_conv: nn.Module = None,
        branch1_norm: nn.Module = None,
        branch2: nn.Module = None,
        activation: nn.Module = None,
        branch_fusion: Callable = None,
    ) -> nn.Module:
        """
        Args:
            branch1_conv (torch.nn.modules): convolutional module in branch1.
            branch1_norm (torch.nn.modules): normalization module in branch1.
            branch2 (torch.nn.modules): bottleneck block module in branch2.
            activation (torch.nn.modules): activation module.
            branch_fusion: (Callable): A callable or layer that combines branch1
                and branch2.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.branch2 is not None

    def forward(self, x) -> torch.Tensor:
        if self.branch1_conv is None:
            x = self.branch_fusion(x, self.branch2(x))
        else:
            shortcut = self.branch1_conv(x)
            if self.branch1_norm is not None:
                shortcut = self.branch1_norm(shortcut)
            x = self.branch_fusion(shortcut, self.branch2(x))
        if self.activation is not None:
            x = self.activation(x)
        return x

class BottleneckBlock(nn.Module):
    """
    Bottleneck block: a sequence of spatiotemporal Convolution, Normalization,
    and Activations repeated in the following order:

    ::


                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                                    Conv3d (conv_b)
                                           ↓
                                 Normalization (norm_b)
                                           ↓
                                   Activation (act_b)
                                           ↓
                                    Conv3d (conv_c)
                                           ↓
                                 Normalization (norm_c)

    The builder can be found in `create_bottleneck_block`.
    """

    def __init__(
        self,
        *,
        conv_a: nn.Module = None,
        norm_a: nn.Module = None,
        act_a: nn.Module = None,
        conv_b: nn.Module = None,
        norm_b: nn.Module = None,
        act_b: nn.Module = None,
        conv_c: nn.Module = None,
        norm_c: nn.Module = None,
    ) -> None:
        """
        Args:
            conv_a (torch.nn.modules): convolutional module.
            norm_a (torch.nn.modules): normalization module.
            act_a (torch.nn.modules): activation module.
            conv_b (torch.nn.modules): convolutional module.
            norm_b (torch.nn.modules): normalization module.
            act_b (torch.nn.modules): activation module.
            conv_c (torch.nn.modules): convolutional module.
            norm_c (torch.nn.modules): normalization module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert all(op is not None for op in (self.conv_a, self.conv_b, self.conv_c))
        if self.norm_c is not None:
            # This flag is used for weight initialization.
            self.norm_c.block_final_bn = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicitly forward every layer.
        # Branch2a, for example Tx1x1, BN, ReLU.
        x = self.conv_a(x)
        if self.norm_a is not None:
            x = self.norm_a(x)
        if self.act_a is not None:
            x = self.act_a(x)

        # Branch2b, for example 1xHxW, BN, ReLU.
        x = self.conv_b(x)
        if self.norm_b is not None:
            x = self.norm_b(x)
        if self.act_b is not None:
            x = self.act_b(x)

        # Branch2c, for example 1x1x1, BN.
        x = self.conv_c(x)
        if self.norm_c is not None:
            x = self.norm_c(x)
        return x

class ResStage(nn.Module):
    """
    ResStage composes sequential blocks that make up a ResNet. These blocks could be,
    for example, Residual blocks, Non-Local layers, or Squeeze-Excitation layers.

    ::


                                        Input
                                           ↓
                                       ResBlock
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                       ResBlock

    The builder can be found in `create_res_stage`.
    """

    def __init__(self, res_blocks: nn.ModuleList) -> nn.Module:
        """
        Args:
            res_blocks (torch.nn.module_list): ResBlock module(s).
        """
        super().__init__()
        self.res_blocks = res_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _, res_block in enumerate(self.res_blocks):
            x = res_block(x)
        return x

class ResNetBasicStem(nn.Module):
    """
    ResNet basic 3D stem module. Performs spatiotemporal Convolution, BN, and activation
    following by a spatiotemporal pooling.

    ::

                                        Conv3d
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                                        Pool3d

    The builder can be found in `create_res_basic_stem`.
    """

    def __init__(
        self,
        *,
        conv: nn.Module = None,
        norm: nn.Module = None,
        activation: nn.Module = None,
        pool: nn.Module = None,
    ) -> None:
        """
        Args:
            conv (torch.nn.modules): convolutional module.
            norm (torch.nn.modules): normalization module.
            activation (torch.nn.modules): activation module.
            pool (torch.nn.modules): pooling module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.conv is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

def create_x3d_stem(
    *,
    # Conv configs.
    in_channels: int,
    out_channels: int,
    conv_kernel_size: Tuple[int] = (5, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    conv_padding: Tuple[int] = (2, 1, 1),
    # BN configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
) -> nn.Module:
    """
    Creates the stem layer for X3D. It performs spatial Conv, temporal Conv, BN, and Relu.

    ::

                                        Conv_xy
                                           ↓
                                        Conv_t
                                           ↓
                                     Normalization
                                           ↓
                                       Activation

    Args:
        in_channels (int): input channel size of the convolution.
        out_channels (int): output channel size of the convolution.
        conv_kernel_size (tuple): convolutional kernel size(s).
        conv_stride (tuple): convolutional stride size(s).
        conv_padding (tuple): convolutional padding size(s).

        norm (callable): a callable that constructs normalization layer, options
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.

        activation (callable): a callable that constructs activation layer, options
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).

    Returns:
        (nn.Module): X3D stem layer.
    """
    conv_xy_module = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(1, conv_kernel_size[1], conv_kernel_size[2]),
        stride=(1, conv_stride[1], conv_stride[2]),
        padding=(0, conv_padding[1], conv_padding[2]),
        bias=False,
    )
    conv_t_module = nn.Conv3d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=(conv_kernel_size[0], 1, 1),
        stride=(conv_stride[0], 1, 1),
        padding=(conv_padding[0], 0, 0),
        bias=False,
        groups=out_channels,
    )
    stacked_conv_module = Conv2plus1d(
        conv_t=conv_xy_module,
        norm=None,
        activation=None,
        conv_xy=conv_t_module,
    )

    norm_module = (
        None
        if norm is None
        else norm(num_features=out_channels, eps=norm_eps, momentum=norm_momentum)
    )
    activation_module = None if activation is None else activation()

    return ResNetBasicStem(
        conv=stacked_conv_module,
        norm=norm_module,
        activation=activation_module,
        pool=None,
    )

def create_x3d_bottleneck_block(
    *,
    # Convolution configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    conv_kernel_size: Tuple[int] = (3, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    se_ratio: float = 0.0625,
    # Activation configs.
    activation: Callable = nn.ReLU,
    inner_act: Callable = Swish,
) -> nn.Module:
    """
    Bottleneck block for X3D: a sequence of Conv, Normalization with optional SE block,
    and Activations repeated in the following order:

    ::

                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                                    Conv3d (conv_b)
                                           ↓
                                 Normalization (norm_b)
                                           ↓
                                 Squeeze-and-Excitation
                                           ↓
                                   Activation (act_b)
                                           ↓
                                    Conv3d (conv_c)
                                           ↓
                                 Normalization (norm_c)

    Args:
        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        conv_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_stride (tuple): convolutional stride size(s) for conv_b.

        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        se_ratio (float): if > 0, apply SE to the 3x3x3 conv, with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.

        activation (callable): a callable that constructs activation layer, examples
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).
        inner_act (callable): whether use Swish activation for act_b or not.

    Returns:
        (nn.Module): X3D bottleneck block.
    """
    # 1x1x1 Conv
    conv_a = nn.Conv3d(
        in_channels=dim_in, out_channels=dim_inner, kernel_size=(1, 1, 1), bias=False
    )
    norm_a = (
        None
        if norm is None
        else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    )
    act_a = None if activation is None else activation()

    # 3x3x3 Conv
    conv_b = nn.Conv3d(
        in_channels=dim_inner,
        out_channels=dim_inner,
        kernel_size=conv_kernel_size,
        stride=conv_stride,
        padding=[size // 2 for size in conv_kernel_size],
        bias=False,
        groups=dim_inner,
        dilation=(1, 1, 1),
    )
    se = (
        SqueezeExcitation(
            num_channels=dim_inner,
            num_channels_reduced=round_width(dim_inner, se_ratio),
            is_3d=True,
        )
        if se_ratio > 0.0
        else nn.Identity()
    )
    norm_b = nn.Sequential(
        (
            nn.Identity()
            if norm is None
            else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
        ),
        se,
    )
    act_b = None if inner_act is None else inner_act()

    # 1x1x1 Conv
    conv_c = nn.Conv3d(
        in_channels=dim_inner, out_channels=dim_out, kernel_size=(1, 1, 1), bias=False
    )
    norm_c = (
        None
        if norm is None
        else norm(num_features=dim_out, eps=norm_eps, momentum=norm_momentum)
    )

    return BottleneckBlock(
        conv_a=conv_a,
        norm_a=norm_a,
        act_a=act_a,
        conv_b=conv_b,
        norm_b=norm_b,
        act_b=act_b,
        conv_c=conv_c,
        norm_c=norm_c,
    )

def create_x3d_res_block(
    *,
    # Bottleneck Block configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    bottleneck: Callable = create_x3d_bottleneck_block,
    use_shortcut: bool = True,
    # Conv configs.
    conv_kernel_size: Tuple[int] = (3, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    se_ratio: float = 0.0625,
    # Activation configs.
    activation: Callable = nn.ReLU,
    inner_act: Callable = Swish,
) -> nn.Module:
    """
    Residual block for X3D. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.

    ::

                                         Input
                                           |-------+
                                           ↓       |
                                         Block     |
                                           ↓       |
                                       Summation ←-+
                                           ↓
                                       Activation

    Args:
        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        bottleneck (callable): a callable for create_x3d_bottleneck_block.

        conv_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_stride (tuple): convolutional stride size(s) for conv_b.

        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        se_ratio (float): if > 0, apply SE to the 3x3x3 conv, with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.

        activation (callable): a callable that constructs activation layer, examples
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).
        inner_act (callable): whether use Swish activation for act_b or not.

    Returns:
        (nn.Module): X3D block layer.
    """

    norm_model = None
    if norm is not None and dim_in != dim_out:
        norm_model = norm(num_features=dim_out)

    return ResBlock(
        branch1_conv=nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=conv_stride,
            bias=False,
        )
        if (dim_in != dim_out or np.prod(conv_stride) > 1) and use_shortcut
        else None,
        branch1_norm=norm_model if dim_in != dim_out and use_shortcut else None,
        branch2=bottleneck(
            dim_in=dim_in,
            dim_inner=dim_inner,
            dim_out=dim_out,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            se_ratio=se_ratio,
            activation=activation,
            inner_act=inner_act,
        ),
        activation=None if activation is None else activation(),
        branch_fusion=lambda x, y: x + y,
    )

def create_x3d_res_stage(
    *,
    # Stage configs.
    depth: int,
    # Bottleneck Block configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    bottleneck: Callable = create_x3d_bottleneck_block,
    # Conv configs.
    conv_kernel_size: Tuple[int] = (3, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    se_ratio: float = 0.0625,
    # Activation configs.
    activation: Callable = nn.ReLU,
    inner_act: Callable = Swish,
) -> nn.Module:
    """
    Create Residual Stage, which composes sequential blocks that make up X3D.

    ::

                                        Input
                                           ↓
                                       ResBlock
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                       ResBlock

    Args:

        depth (init): number of blocks to create.

        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        bottleneck (callable): a callable for create_x3d_bottleneck_block.

        conv_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_stride (tuple): convolutional stride size(s) for conv_b.

        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        se_ratio (float): if > 0, apply SE to the 3x3x3 conv, with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.

        activation (callable): a callable that constructs activation layer, examples
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).
        inner_act (callable): whether use Swish activation for act_b or not.

    Returns:
        (nn.Module): X3D stage layer.
    """
    res_blocks = []
    for idx in range(depth):
        block = create_x3d_res_block(
            dim_in=dim_in if idx == 0 else dim_out,
            dim_inner=dim_inner,
            dim_out=dim_out,
            bottleneck=bottleneck,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride if idx == 0 else (1, 1, 1),
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            se_ratio=(se_ratio if (idx + 1) % 2 else 0.0),
            activation=activation,
            inner_act=inner_act,
        )
        res_blocks.append(block)

    return ResStage(res_blocks=nn.ModuleList(res_blocks))

def create_x3d_head(
    *,
    # Projection configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    num_classes: int,
    # Pooling configs.
    pool_act: Callable = nn.ReLU,
    pool_kernel_size: Tuple[int] = (13, 5, 5),
    # BN configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    bn_lin5_on=False,
    # Dropout configs.
    dropout_rate: float = 0.5,
    # Activation configs.
    activation: Callable = nn.Softmax,
    # Output configs.
    output_with_global_average: bool = True,
) -> nn.Module:
    """
    Creates X3D head. This layer performs an projected pooling operation followed
    by an dropout, a fully-connected projection, an activation layer and a global
    spatiotemporal averaging.

    ::

                                     ProjectedPool
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
                                           ↓
                                       Averaging

    Args:
        dim_in (int): input channel size of the X3D head.
        dim_inner (int): intermediate channel size of the X3D head.
        dim_out (int): output channel size of the X3D head.
        num_classes (int): the number of classes for the video dataset.

        pool_act (callable): a callable that constructs resnet pool activation
            layer such as nn.ReLU.
        pool_kernel_size (tuple): pooling kernel size(s) when not using adaptive
            pooling.

        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        bn_lin5_on (bool): if True, perform normalization on the features
            before the classifier.

        dropout_rate (float): dropout rate.

        activation (callable): a callable that constructs resnet head activation
            layer, examples include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not
            applying activation).

        output_with_global_average (bool): if True, perform global averaging on temporal
            and spatial dimensions and reshape output to batch_size x out_features.

    Returns:
        (nn.Module): X3D head layer.
    """
    pre_conv_module = nn.Conv3d(
        in_channels=dim_in, out_channels=dim_inner, kernel_size=(1, 1, 1), bias=False
    )

    pre_norm_module = norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    pre_act_module = None if pool_act is None else pool_act()

    if pool_kernel_size is None:
        pool_module = nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        pool_module = nn.AvgPool3d(pool_kernel_size, stride=1)

    post_conv_module = nn.Conv3d(
        in_channels=dim_inner, out_channels=dim_out, kernel_size=(1, 1, 1), bias=False
    )

    if bn_lin5_on:
        post_norm_module = norm(
            num_features=dim_out, eps=norm_eps, momentum=norm_momentum
        )
    else:
        post_norm_module = None
    post_act_module = None if pool_act is None else pool_act()

    projected_pool_module = ProjectedPool(
        pre_conv=pre_conv_module,
        pre_norm=pre_norm_module,
        pre_act=pre_act_module,
        pool=pool_module,
        post_conv=post_conv_module,
        post_norm=post_norm_module,
        post_act=post_act_module,
    )

    if activation is None:
        activation_module = None
    elif activation == nn.Softmax:
        activation_module = activation(dim=1)
    elif activation == nn.Sigmoid:
        activation_module = activation()
    else:
        raise NotImplementedError(
            "{} is not supported as an activation" "function.".format(activation)
        )

    if output_with_global_average:
        output_pool = nn.AdaptiveAvgPool3d(1)
    else:
        output_pool = None

    return ResNetBasicHead(
        proj=nn.Linear(dim_out, num_classes, bias=True),
        activation=activation_module,
        pool=projected_pool_module,
        dropout=nn.Dropout(dropout_rate) if dropout_rate > 0 else None,
        output_pool=output_pool,
    )

def create_x3d(
    *,
    # Input clip configs.
    input_channel: int = 3,
    input_clip_length: int = 13,
    input_crop_size: int = 160,
    # Model configs.
    model_num_class: int = 400,
    dropout_rate: float = 0.5,
    width_factor: float = 2.0,
    depth_factor: float = 2.2,
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_in: int = 12,
    stem_conv_kernel_size: Tuple[int] = (5, 3, 3),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    # Stage configs.
    stage_conv_kernel_size: Tuple[Tuple[int]] = (
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    ),
    stage_spatial_stride: Tuple[int] = (2, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
    bottleneck: Callable = create_x3d_bottleneck_block,
    bottleneck_factor: float = 2.25,
    se_ratio: float = 0.0625,
    inner_act: Callable = Swish,
    # Head configs.
    head_dim_out: int = 2048,
    head_pool_act: Callable = nn.ReLU,
    head_bn_lin5_on: bool = False,
    head_activation: Callable = None,
    head_output_with_global_average: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730

    ::

                                         Input
                                           ↓
                                         Stem
                                           ↓
                                         Stage 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Stage N
                                           ↓
                                         Head

    Args:
        input_channel (int): number of channels for the input video clip.
        input_clip_length (int): length of the input video clip. Value for
            different models: X3D-XS: 4; X3D-S: 13; X3D-M: 16; X3D-L: 16.
        input_crop_size (int): spatial resolution of the input video clip.
            Value for different models: X3D-XS: 160; X3D-S: 160; X3D-M: 224;
            X3D-L: 312.

        model_num_class (int): the number of classes for the video dataset.
        dropout_rate (float): dropout rate.
        width_factor (float): width expansion factor.
        depth_factor (float): depth expansion factor. Value for different
            models: X3D-XS: 2.2; X3D-S: 2.2; X3D-M: 2.2; X3D-L: 5.0.

        norm (callable): a callable that constructs normalization layer.
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.

        activation (callable): a callable that constructs activation layer.

        stem_dim_in (int): input channel size for stem before expansion.
        stem_conv_kernel_size (tuple): convolutional kernel size(s) of stem.
        stem_conv_stride (tuple): convolutional stride size(s) of stem.

        stage_conv_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        stage_spatial_stride (tuple): the spatial stride for each stage.
        stage_temporal_stride (tuple): the temporal stride for each stage.
        bottleneck_factor (float): bottleneck expansion factor for the 3x3x3 conv.
        se_ratio (float): if > 0, apply SE to the 3x3x3 conv, with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.
        inner_act (callable): whether use Swish activation for act_b or not.

        head_dim_out (int): output channel size of the X3D head.
        head_pool_act (callable): a callable that constructs resnet pool activation
            layer such as nn.ReLU.
        head_bn_lin5_on (bool): if True, perform normalization on the features
            before the classifier.
        head_activation (callable): a callable that constructs activation layer.
        head_output_with_global_average (bool): if True, perform global averaging on
            the head output.

    Returns:
        (nn.Module): the X3D network.
    """

    blocks = []
    # Create stem for X3D.
    stem_dim_out = round_width(stem_dim_in, width_factor)
    stem = create_x3d_stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        norm=norm,
        norm_eps=norm_eps,
        norm_momentum=norm_momentum,
        activation=activation,
    )
    blocks.append(stem)

    # Compute the depth and dimension for each stage
    stage_depths = [1, 2, 5, 3]
    exp_stage = 2.0
    stage_dim1 = stem_dim_in
    stage_dim2 = round_width(stage_dim1, exp_stage, divisor=8)
    stage_dim3 = round_width(stage_dim2, exp_stage, divisor=8)
    stage_dim4 = round_width(stage_dim3, exp_stage, divisor=8)
    stage_dims = [stage_dim1, stage_dim2, stage_dim3, stage_dim4]

    dim_in = stem_dim_out
    # Create each stage for X3D.
    for idx in range(len(stage_depths)):
        dim_out = round_width(stage_dims[idx], width_factor)
        dim_inner = int(bottleneck_factor * dim_out)
        depth = round_repeats(stage_depths[idx], depth_factor)

        stage_conv_stride = (
            stage_temporal_stride[idx],
            stage_spatial_stride[idx],
            stage_spatial_stride[idx],
        )

        stage = create_x3d_res_stage(
            depth=depth,
            dim_in=dim_in,
            dim_inner=dim_inner,
            dim_out=dim_out,
            bottleneck=bottleneck,
            conv_kernel_size=stage_conv_kernel_size[idx],
            conv_stride=stage_conv_stride,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            se_ratio=se_ratio,
            activation=activation,
            inner_act=inner_act,
        )
        blocks.append(stage)
        dim_in = dim_out

    # # Create head for X3D.
    # total_spatial_stride = stem_conv_stride[1] * np.prod(stage_spatial_stride)
    # total_temporal_stride = stem_conv_stride[0] * np.prod(stage_temporal_stride)

    # assert (
    #     input_clip_length >= total_temporal_stride
    # ), "Clip length doesn't match temporal stride!"
    # assert (
    #     input_crop_size >= total_spatial_stride
    # ), "Crop size doesn't match spatial stride!"

    # head_pool_kernel_size = (
    #     input_clip_length // total_temporal_stride,
    #     int(math.ceil(input_crop_size / total_spatial_stride)),
    #     int(math.ceil(input_crop_size / total_spatial_stride)),
    # )

    # head = create_x3d_head(
    #     dim_in=dim_out,
    #     dim_inner=dim_inner,
    #     dim_out=head_dim_out,
    #     num_classes=model_num_class,
    #     pool_act=head_pool_act,
    #     pool_kernel_size=head_pool_kernel_size,
    #     norm=norm,
    #     norm_eps=norm_eps,
    #     norm_momentum=norm_momentum,
    #     bn_lin5_on=head_bn_lin5_on,
    #     dropout_rate=dropout_rate,
    #     activation=head_activation,
    #     output_with_global_average=head_output_with_global_average,
    # )
    # blocks.append(head)
    return Net(blocks=nn.ModuleList(blocks))

class ProjectedPool(nn.Module):
    """
    A pooling module augmented with Conv, Normalization and Activation both
    before and after pooling for the head layer of X3D.

    ::

                                    Conv3d (pre_conv)
                                           ↓
                                 Normalization (pre_norm)
                                           ↓
                                   Activation (pre_act)
                                           ↓
                                        Pool3d
                                           ↓
                                    Conv3d (post_conv)
                                           ↓
                                 Normalization (post_norm)
                                           ↓
                                   Activation (post_act)
    """

    def __init__(
        self,
        *,
        pre_conv: nn.Module = None,
        pre_norm: nn.Module = None,
        pre_act: nn.Module = None,
        pool: nn.Module = None,
        post_conv: nn.Module = None,
        post_norm: nn.Module = None,
        post_act: nn.Module = None,
    ) -> None:
        """
        Args:
            pre_conv (torch.nn.modules): convolutional module.
            pre_norm (torch.nn.modules): normalization module.
            pre_act (torch.nn.modules): activation module.
            pool (torch.nn.modules): pooling module.
            post_conv (torch.nn.modules): convolutional module.
            post_norm (torch.nn.modules): normalization module.
            post_act (torch.nn.modules): activation module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.pre_conv is not None
        assert self.pool is not None
        assert self.post_conv is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(x)

        if self.pre_norm is not None:
            x = self.pre_norm(x)
        if self.pre_act is not None:
            x = self.pre_act(x)

        x = self.pool(x)
        x = self.post_conv(x)

        if self.post_norm is not None:
            x = self.post_norm(x)
        if self.post_act is not None:
            x = self.post_act(x)
        return x

def _x3d(
    checkpoint_path: str = None,
    **kwargs: Any,
) -> nn.Module:
    model = create_x3d(**kwargs)
    if checkpoint_path is not None:
        # All models are loaded onto CPU by default
        print("Load pretrained backbone weight from :", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict, strict=False)
    return model


configs = {
    'x3d_xs': 
            dict(input_clip_length=4,
            input_crop_size=160,
            width_factor = 1.5,
            num_channels = [24,48,96,192]
                        ),
    'x3d_s': 
            dict(input_clip_length=5,
            width_factor = 2,
            num_channels = [24,48,96,192]
            
                        ),
    'x3d_m': 
            dict(input_clip_length=16,
            input_crop_size=224,
            num_channels = [24,48,96,192]
                        ),
    'x3d_l': 
            dict(input_clip_length=16,
            input_crop_size=312,
            depth_factor=5.0,
            num_channels = [24,48,96,192]
                        ),
    'x3d_self': 
            dict(input_clip_length=5,
            width_factor = 2,
            num_channels = [24,48,96,192]
                        )
}

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, strides=[4, 8, 16, 32], num_channels=[96, 192, 384, 768]):
        super().__init__()
        self.strides = strides
        self.num_channels = num_channels
        self.body = backbone

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            # print(x.shape)
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            out[name] = NestedTensor(x, mask)
        # assert False
        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 pretrained: str = None):
        assert name in ['x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l', 'x3d_self']
        print("Using backbone: ", name)
        cfgs = configs[name]
        cfgs.update({'checkpoint_path': pretrained})
        out_indices = (0, 1, 2, 3)
        strides = [int(2**(i+2)) for i in out_indices]
        # num_channels = [int(8 * (2 + i)) for i in out_indices]
        num_channels = cfgs['num_channels']
        backbone = _x3d(**cfgs)
        super().__init__(backbone, strides, num_channels)



class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels


    def forward(self, tensor_list: NestedTensor):
        tensor_list.tensors = rearrange(tensor_list.tensors, 'b t c h w -> b c t h w')
        tensor_list.mask = rearrange(tensor_list.mask, 'b t h w -> (b t) h w')

        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():

            if name == '0':
                continue
            out.append(x)
            # position encoding

            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_x3d_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone(args.backbone, args.backbone_pretrained)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

if __name__ == '__main__':
    cfgs = configs['x3d_l']
    # model = torch.hub.load('facebookresearch/pytorchvideo', "x3d_l", pretrained=True)
    # model = _x3d(checkpoint_path='/home/xhu/Code/narval_ref/ckp/X3D_XS.pyth',input_clip_length=4,input_crop_size=160)
    # model = _x3d(checkpoint_path='/home/xhu/Code/narval_ref/ckp/X3D_L.pyth', **cfgs)
    # model = _x3d(input_clip_length=5,input_crop_size=312)
    model = Backbone(name = 'x3d_xs')
    # model.load_state_dict(torch.load('/home/xhu/.cache/torch/hub/checkpoints/X3D_XS.pyth')['model_state'])
    inputs = NestedTensor(torch.randn(1, 4, 3, 416, 640),torch.randn(1, 4, 416, 640)) # 10 = 2 x 5
    # import ipdb; ipdb.set_trace()
    # outs
    # 0: (10, 96, 96, 56)
    # 1: (10, 192, 48, 28)
    # 2: (10, 384, 24, 14)
    # 3: (10, 768, 12, 7)
    out = model(inputs)
    for _,layer in out.items():
        print(_,":",layer.tensors.shape, layer.mask.shape)