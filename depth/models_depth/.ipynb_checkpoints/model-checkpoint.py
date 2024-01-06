# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The deconvolution code is based on Simple Baseline.
# (https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer, constant_init, normal_init
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from meta_prompts.models import UNetWrapper, TextAdapterDepth

class MetaPromptDepthEncoder(nn.Module):
    def __init__(self, args, out_dim=1024, ldm_prior=[320, 640, 1280, 1280], sd_path=None, text_dim=768, 
                 dataset='kitti', num_prompt=50
                 ):
        super().__init__()
        config = OmegaConf.load('./v1-inference.yaml')
        if sd_path is None:
            config.model.params.ckpt_path = '../checkpoints/v1-5-pruned-emaonly.ckpt'
        else:
            config.model.params.ckpt_path = f'../{sd_path}'

        self.layer1 = nn.Sequential(
        nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=1, padding=1),
        nn.GroupNorm(16, ldm_prior[0]),
        nn.ReLU(),
        )
        self.layer2 = Decoder(ldm_prior[1], ldm_prior[1], 1, [32], [4])
        self.layer3 = Decoder(ldm_prior[2], ldm_prior[2], 2, [32, 32], [4, 4])
        self.layer4 = Decoder(ldm_prior[3], ldm_prior[3], 3, [32, 32, 32], [4, 4, 4])

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), text_dim, 3, 1, 1),
            nn.GroupNorm(16, text_dim),
            nn.ReLU(),
        )
        self.apply(self._init_weights)

        sd_model = instantiate_from_config(config.model)
        self.refine_step = args.refine_step
        self.num_prompt = num_prompt
        self.share_meta_prompts = False
        print(f'Set Unet refine step: {self.refine_step}, Set query number: {num_prompt}')

        for i in range(self.refine_step):
            if i > 0:
                cross_unet_conv =  sd_model.model.diffusion_model.out
                setattr(self, f"cross_unet_conv{i + 1}", cross_unet_conv)
            t = nn.Parameter(torch.randn(1, 1280), requires_grad=True)
            setattr(self, f"t{i + 1}", t)
            if not self.share_meta_prompts:
                meta_prompts = nn.Parameter(torch.randn(num_prompt, text_dim), requires_grad=True)
                setattr(self, f"meta_prompts{i + 1}", meta_prompts)
                text_adapter = TextAdapterDepth()
                setattr(self, f"text_adapter{i + 1}", text_adapter)
        if self.share_meta_prompts:
            self.meta_prompts = nn.Parameter(torch.randn(num_prompt, text_dim), requires_grad=True)
        
        ### stable diffusion layers 
        self.encoder_vq = sd_model.first_stage_model
        self.unet = UNetWrapper(sd_model.model, use_attn=False)
        del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        del self.unet.unet.diffusion_model.time_embed

        for param in self.encoder_vq.parameters():
            param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        img = x
        with torch.no_grad():
            latents = self.encoder_vq.encode(x).mode()
            latents = latents.detach()  
        outs = []
        for i in range(self.refine_step):
            if isinstance(latents, list):
                latents = latents[0]
            if i > 0:
                cross_unet_conv = getattr(self, f"cross_unet_conv{i + 1}")
                latents = cross_unet_conv(latents)
            if not self.share_meta_prompts:
                meta_prompts = getattr(self, f"meta_prompts{i + 1}")
                # c_crossattn = meta_prompts[None, :, :].expand(x.shape[0], -1, -1)
                text_adapter = getattr(self, f"text_adapter{i + 1}")
                c_crossattn = text_adapter(latents, meta_prompts) 
            else:
                c_crossattn = self.meta_prompts[None, :, :].expand(x.shape[0], -1, -1)
            t = getattr(self, f"t{i + 1}")
            t = t.repeat(x.shape[0], 1)
            latents = self.unet(latents, t, c_crossattn=[c_crossattn])  
            outs.append(latents)
        outs = outs[-1]
        x = torch.cat([self.layer1(outs[0]), self.layer2([outs[1]]), self.layer3([outs[2]]), self.layer4([outs[3]])], dim=1)
        out = self.out_layer(x)
        out = torch.einsum('bchw,bnc->bnhw', out, c_crossattn)
        return out

class MetaPromptDepth(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.max_depth = args.max_depth

        embed_dim = 192
        channels_in = embed_dim*8
        channels_out = embed_dim
        self.resize_scale = args.resize_scale
        num_prompt = 50

        self.encoder = MetaPromptDepthEncoder(args, out_dim=channels_in, dataset=args.dataset, num_prompt=num_prompt)
        num_deconv, num_filters, deconv_kernels = args.num_deconv, args.num_filters, args.deconv_kernels
        self.decoder = Decoder(num_prompt, channels_out, num_deconv, num_filters, deconv_kernels)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, x):    
        b, c, h, w = x.shape
        x = x*2.0 - 1.0  # normalize to [-1, 1]
        # x = x[:, :, 4:, 4:]
        res_shape = ((h // 64 + self.resize_scale) * 64, (w // 64 + self.resize_scale) * 64)
        x = F.interpolate(x, size=res_shape, mode='bilinear', align_corners=False)

        conv_feats = self.encoder(x)
        conv_feats = F.interpolate(conv_feats, size=(h//8, w//8), mode='bilinear', align_corners=False)

        out = self.decoder([conv_feats])
        out = self.last_layer_depth(out)
        out_depth = torch.sigmoid(out) * self.max_depth
        out_dict = {'pred_d': out_depth}

        return out_dict


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_deconv, num_filters, deconv_kernels):
        super().__init__()
        self.deconv = num_deconv
        self.in_channels = in_channels

        self.deconv_layers = self._make_deconv_layer(
            num_deconv,
            num_filters,
            deconv_kernels,
        )
        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
        out = self.deconv_layers(conv_feats[0])
        out = self.conv_layers(out)
        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

