# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)

# from mmseg.registry import MODELS


class Mlp(BaseModule):
    """Multi Layer Perceptron (MLP) Module.

    Args:
        in_features (int): The dimension of input features.
        hidden_features (int): The dimension of hidden features.
            Defaults: None.
        out_features (int): The dimension of output features.
            Defaults: None.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=True,
            groups=hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward function."""

        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(BaseModule):
    """Stem Block at the beginning of Semantic Branch.

    Args:
        in_channels (int): The dimension of input channels.
        out_channels (int): The dimension of output channels.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            build_activation_layer(act_cfg),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        """Forward function."""

        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class MSCAAttention(BaseModule):
    """Attention Module in Multi-Scale Convolutional Attention Module (MSCA).

    Args:
        channels (int): The dimension of channels.
        kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
    """

    def __init__(self,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """Forward function."""

        u = x.clone()

        attn = self.conv0(x)

        # Multi-Scale Feature extraction
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        # Channel Mixing
        attn = self.conv3(attn)

        # Convolutional Attention
        x = attn * u

        return x


class MSCASpatialAttention(BaseModule):
    """Spatial Attention Module in Multi-Scale Convolutional Attention Module
    (MSCA).

    Args:
        in_channels (int): The dimension of channels.
        attention_kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
    """

    def __init__(self,
                 in_channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = build_activation_layer(act_cfg)
        self.spatial_gating_unit = MSCAAttention(in_channels,
                                                 attention_kernel_sizes,
                                                 attention_kernel_paddings)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        """Forward function."""

        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class MSCABlock(BaseModule):
    """Basic Multi-Scale Convolutional Attention Block. It leverage the large-
    kernel attention (LKA) mechanism to build both channel and spatial
    attention. In each branch, it uses two depth-wise strip convolutions to
    approximate standard depth-wise convolutions with large kernels. The kernel
    size for each branch is set to 7, 11, and 21, respectively.

    Args:
        channels (int): The dimension of channels.
        attention_kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        mlp_ratio (float): The ratio of multiple input dimension to
            calculate hidden feature in MLP layer. Defaults: 4.0.
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
        drop_path (float): The ratio of drop paths.
            Defaults: 0.0.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.attn = MSCASpatialAttention(channels, attention_kernel_sizes,
                                         attention_kernel_paddings, act_cfg)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        mlp_hidden_channels = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_channels,
            act_cfg=act_cfg,
            drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x, H, W):
        """Forward function."""

        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Args:
        patch_size (int): The patch size.
            Defaults: 7.
        stride (int): Stride of the convolutional layer.
            Default: 4.
        in_channels (int): The number of input channels.
            Defaults: 3.
        embed_dims (int): The dimensions of embedding.
            Defaults: 768.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_channels=3,
                 embed_dim=768,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2)
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        """Forward function."""

        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


# @MODELS.register_module()
class MSCAN(BaseModule):
    """SegNeXt Multi-Scale Convolutional Attention Network (MCSAN) backbone.

    This backbone is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.

    Args:
        in_channels (int): The number of input channels. Defaults: 3.
        embed_dims (list[int]): Embedding dimension.
            Defaults: [64, 128, 256, 512].
        mlp_ratios (list[int]): Ratio of mlp hidden dim to embedding dim.
            Defaults: [4, 4, 4, 4].
        drop_rate (float): Dropout rate. Defaults: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.
        depths (list[int]): Depths of each Swin Transformer stage.
            Default: [3, 4, 6, 3].
        num_stages (int): MSCAN stages. Default: 4.
        attention_kernel_sizes (list): Size of attention kernel in
            Attention Module (Figure 2(b) of original paper).
            Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): Size of attention paddings
            in Attention Module (Figure 2(b) of original paper).
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        norm_cfg (dict): Config of norm layers.
            Defaults: dict(type='SyncBN', requires_grad=True).
        pretrained (str, optional): model pretrained path.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(in_channels, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    norm_cfg=norm_cfg)

            block = nn.ModuleList([
                MSCABlock(
                    channels=embed_dims[i],
                    attention_kernel_sizes=attention_kernel_sizes,
                    attention_kernel_paddings=attention_kernel_paddings,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

    def init_weights(self):
        """Initialize modules of MSCAN."""

        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        """Forward function."""

        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


class MultiScaleChannelAttention(BaseModule):
    """多尺度通道注意力模块
    
    通过在不同空间尺度上计算通道注意力，更好地捕获通道间的重要性关系
    
    Args:
        channels (int): 输入通道数
        scales (list): 多尺度池化的尺度列表，默认[1, 3, 5, 7]
        reduction (int): 通道压缩比例，默认4
        use_global (bool): 是否使用全局上下文增强，默认True
        activation (str): 激活函数类型，默认'ReLU'
    """
    
    def __init__(self, 
                 channels, 
                 scales=[1, 3, 5, 7], 
                 reduction=4, 
                 use_global=True,
                 activation='ReLU'):
        super(MultiScaleChannelAttention, self).__init__()
        
        self.channels = channels
        self.scales = scales
        self.reduction = reduction
        self.use_global = use_global
        
        # 多尺度池化分支
        self.multi_scale_pools = nn.ModuleList()
        for scale in scales:
            if scale == 1:
                # 全局平均池化
                self.multi_scale_pools.append(nn.AdaptiveAvgPool2d(1))
            else:
                # 多尺度平均池化
                self.multi_scale_pools.append(nn.AdaptiveAvgPool2d(scale))
        
        # 特征变换层
        hidden_channels = max(channels // reduction, 8)  # 确保最小通道数
        
        # 每个尺度的特征变换
        self.scale_transforms = nn.ModuleList()
        for _ in scales:
            self.scale_transforms.append(
                nn.Sequential(
                    nn.Conv2d(channels, hidden_channels, 1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    self._get_activation(activation)
                )
            )
        
        # 尺度融合网络
        total_hidden_channels = hidden_channels * len(scales)
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(total_hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            self._get_activation(activation),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
        )
        
        # 全局上下文分支（可选）
        if use_global:
            self.global_context = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // reduction, 1, bias=False),
                self._get_activation(activation),
                nn.Conv2d(channels // reduction, channels, 1, bias=False),
            )
        
        # 最终激活
        self.sigmoid = nn.Sigmoid()
        
        # 可学习的尺度权重
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))
        
        self._init_weights()
    
    def _get_activation(self, activation):
        """获取激活函数"""
        if activation == 'ReLU':
            return nn.ReLU(inplace=True)
        elif activation == 'GELU':
            return nn.GELU()
        elif activation == 'SiLU':
            return nn.SiLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x (Tensor): 输入特征图 [B, C, H, W]
            
        Returns:
            Tensor: 注意力权重 [B, C, 1, 1]
        """
        B, C, H, W = x.size()
        
        # 1. 多尺度特征提取
        scale_features = []
        scale_weights_norm = F.softmax(self.scale_weights, dim=0)
        
        for i, (pool, transform) in enumerate(zip(self.multi_scale_pools, self.scale_transforms)):
            # 多尺度池化
            pooled = pool(x)  # [B, C, scale, scale] or [B, C, 1, 1]
            
            # 特征变换
            transformed = transform(pooled)  # [B, hidden_channels, scale, scale]
            
            # 上采样到统一尺寸 (1x1)
            if transformed.size(2) != 1 or transformed.size(3) != 1:
                transformed = F.adaptive_avg_pool2d(transformed, 1)
            
            # 应用可学习的尺度权重
            transformed = transformed * scale_weights_norm[i]
            scale_features.append(transformed)
        
        # 2. 多尺度特征融合
        fused_features = torch.cat(scale_features, dim=1)  # [B, total_hidden_channels, 1, 1]
        attention = self.scale_fusion(fused_features)  # [B, C, 1, 1]
        
        # 3. 全局上下文增强（可选）
        if self.use_global:
            global_att = self.global_context(x)
            attention = attention + global_att
        
        # 4. 生成最终注意力权重
        attention_weights = self.sigmoid(attention)
        
        return attention_weights


class MultiScaleFusionModule(BaseModule):
    """增强型多尺度特征融合模块
    
    将MSCAN输出的多尺度特征进行融合，并集成多尺度通道注意力机制
    
    Args:
        in_channels_list (list[int]): 输入特征图的通道数列表
        out_channels (int): 输出特征图的通道数
        scales (list[int]): 各特征图的下采样倍数
        attention_scales (list[int]): 通道注意力的多尺度池化尺度
        use_attention (bool): 是否使用多尺度通道注意力
    """
    
    def __init__(self, 
                 in_channels_list=[64, 128, 256, 512], 
                 out_channels=256,
                 scales=[4, 8, 16, 32],
                 attention_scales=[1, 3, 5, 7],
                 use_attention=True):
        super().__init__()
        
        self.scales = scales
        self.use_attention = use_attention
        
        # 1. 特征转换层 - 将不同通道数转换为统一通道数
        self.transform_layers = nn.ModuleList()
        for in_channels in in_channels_list:
            self.transform_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 2. 上采样层 - 将不同分辨率的特征上采样到相同大小
        self.upsample_layers = nn.ModuleList()
        for scale in scales:
            self.upsample_layers.append(
                nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
            )
        
        # 3. 多尺度通道注意力机制 - 学习不同特征的重要性权重
        if use_attention:
            self.channel_attention = MultiScaleChannelAttention(
                channels=out_channels * len(in_channels_list),
                scales=attention_scales,
                reduction=4,
                use_global=True,
                activation='ReLU'
            )
        
        # 4. 特征融合模块 - 自适应融合多尺度特征
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 5. 细化模块 - 进一步优化融合特征
        self.refinement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # 6. 残差连接 - 保留原始特征信息
        self.residual_proj = nn.Conv2d(out_channels * len(in_channels_list), out_channels, 1)
        
        # 7. 最终激活
        self.final_act = nn.ReLU(inplace=True)
    
    def forward(self, features):
        """前向传播
        
        Args:
            features (list[Tensor]): 多尺度特征图列表
            
        Returns:
            Tensor: 融合后的特征图 [B, out_channels, H, W]
        """
        assert len(features) == len(self.transform_layers) == len(self.upsample_layers)
        
        # 1. 特征转换和上采样
        transformed_features = []
        for i, feat in enumerate(features):
            # 转换通道数
            feat = self.transform_layers[i](feat)
            # 上采样到统一分辨率
            feat = self.upsample_layers[i](feat)
            transformed_features.append(feat)
        
        # 2. 特征拼接
        concat_features = torch.cat(transformed_features, dim=1)
        
        # 3. 多尺度通道注意力加权
        if self.use_attention:
            attention_weights = self.channel_attention(concat_features)
            weighted_features = concat_features * attention_weights
        else:
            weighted_features = concat_features
        
        # 4. 残差连接
        residual = self.residual_proj(concat_features)
        
        # 5. 特征融合
        fused_features = self.fusion_conv(weighted_features)
        
        # 6. 特征细化
        refined_features = self.refinement(fused_features)
        
        # 7. 残差连接
        output = self.final_act(refined_features + residual)
        
        return output


class MultiModalRegression(nn.Module):
    def __init__(self, in_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return x


class MCA_STM(nn.Module):
    """多尺度卷积注意力空间-时间模型
    
    使用MSCAN作为骨干网络，结合增强型多尺度特征融合模块，实现高精度下采样
    
    Args:
        in_c (int): 输入通道数
        use_enhanced_fusion (bool): 是否使用增强型融合模块
    """
    def __init__(self, in_c: int) -> None:
        super().__init__()
        # 移除未使用的卷积层
        self.backbone = MSCAN(
            in_channels=in_c, 
            embed_dims=[64, 128, 256, 512], 
            mlp_ratios=[4, 4, 4, 4], 
            depths=[3, 4, 6, 3], 
            num_stages=4, 
            attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]], 
            attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]], 
            act_cfg=dict(type='GELU'), 
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        
    
        # 增强型多尺度特征融合模块（集成多尺度通道注意力）
        self.fusion = MultiScaleFusionModule(
            in_channels_list=[64, 128, 256, 512], 
            out_channels=256, 
            scales=[4, 8, 16, 32],
            attention_scales=[1, 3, 5, 7],  # 多尺度通道注意力的池化尺度
            use_attention=True  # 启用多尺度通道注意力
        )
            
        # 回归模块
        self.regression = MultiModalRegression(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x (torch.Tensor): 输入张量 [B, C, H, W]
            
        Returns:
            torch.Tensor: 输出张量 [B, 1, H, W]
        """
        # 特征提取
        features = self.backbone(x)
        
        # 多尺度特征融合
        fused_features = self.fusion(features)
        
        # 回归预测
        output = self.regression(fused_features)
        
        return output

if __name__ == "__main__":
    # 测试集成多尺度通道注意力的模型
    print("=== 测试集成多尺度通道注意力的MCA_STM模型 ===")
    
    # 创建模型
    model = MCA_STM(in_c=8)
    
    # 测试输入
    x = torch.randn(2, 8, 64, 64)  # 减小batch size以节省内存
    
    print(f"输入形状: {x.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 前向传播测试
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {output.shape}")
    
    # 测试多尺度通道注意力模块
    print("\n=== 测试多尺度通道注意力模块 ===")
    attention_module = MultiScaleChannelAttention(
        channels=1024,  # 4个特征图拼接后的通道数 (256*4)
        scales=[1, 3, 5, 7],
        reduction=4
    )
    
    test_input = torch.randn(2, 1024, 16, 16)
    attention_weights = attention_module(test_input)
    print(f"注意力输入形状: {test_input.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"注意力权重范围: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
    
    print("\n=== 多尺度通道注意力集成成功! ===")
    print("主要创新点:")
    print("1. 多尺度池化 (1x1, 3x3, 5x5, 7x7) 捕获不同感受野的通道重要性")
    print("2. 可学习的尺度权重自适应调节各尺度贡献")
    print("3. 全局上下文增强保留重要的全局信息")
    print("4. 残差连接和特征细化提升融合效果")