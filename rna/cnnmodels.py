import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import BaseNeuralNet
from .modules import PCBActiv
from .modules import upscale_by_reshape_2d


class NormalNet(BaseNeuralNet):
    def __init__(self, data_augment=True):
        super().__init__()
        self.data_augment = data_augment

        # Normals branch
        self.enc_n_1 = PCBActiv(3, 16, activ='leaky', sample='down-3') # 512 -> 256; skip out
        self.enc_n_2 = PCBActiv(16, 64, activ='leaky', sample='down-3') # 256 -> 128; skip out
        self.enc_n_3 = PCBActiv(64, 128, activ='leaky')
        self.enc_n_4 = PCBActiv(128, 256, activ='leaky', sample='down-5') # 128 -> 64
        self.enc_n_5 = PCBActiv(256, 256, activ='leaky')
        self.enc_n_6 = PCBActiv(256 // 4 + 64, 128, activ='leaky') # 64 -> 128; skip in
        self.enc_n_7 = PCBActiv(128 // 4 + 16, 64, activ='leaky') # 128 -> 256; skip in
        self.enc_n_8 = PCBActiv(64 // 4 + 3, 3, bn=False, activ=None, conv_bias=True) 

    def forward(self, normals, mask):
        """
        # Arguments
            normals: float tensor with shape (B, 3, H, W)
            mask: float tensor with shape (B, 1, H, W)

        # Returns
            normals_out: float tensor with shape (B, 3, H, W)
            mask_out: float tensor with shape (B, 1, H, W)
        """

        # Training data augmentation
        if self.training and self.data_augment:
            mask = F.dropout(mask, p=0.1).gt(0.5).float()
            nnoise = torch.zeros_like(normals).normal_(0, 0.5)
            nnoise_mask = F.dropout(mask, p=0.99).gt(0.5).float()
            normals = torch.clamp(normals + nnoise_mask * nnoise, -1, 1)

        mask = mask.tile(1, normals.size(1), 1, 1)

        n_skip1, nm_skip1 = self.enc_n_1(normals, mask)
        n_skip2, nm_skip2 = self.enc_n_2(n_skip1, nm_skip1)
        n, nm = self.enc_n_3(n_skip2, nm_skip2)
        n, nm = self.enc_n_4(n, nm)
        n, nm = self.enc_n_5(n, nm)
        n = upscale_by_reshape_2d(n, 2)
        nm = upscale_by_reshape_2d(nm, 2)
        # skip2 in
        n = torch.cat([n, n_skip2], dim=1)
        nm = torch.cat([nm, nm_skip2], dim=1)
        n, nm = self.enc_n_6(n, nm)
        n = upscale_by_reshape_2d(n, 2)
        nm = upscale_by_reshape_2d(nm, 2)
        # skip1 in
        n = torch.cat([n, n_skip1], dim=1)
        nm = torch.cat([nm, nm_skip1], dim=1)
        n, nm = self.enc_n_7(n, nm)
        n = upscale_by_reshape_2d(n, 2)
        nm = upscale_by_reshape_2d(nm, 2)
        # skip input
        n = torch.cat([n, normals], dim=1)
        nm = torch.cat([nm, mask], dim=1)
        normals_out, mask_out = self.enc_n_8(n, nm)

        return normals_out, mask_out[:, :1]


class VisibilityNet(BaseNeuralNet):
    def __init__(self, data_augment=True):
        super().__init__()
        self.data_augment = data_augment

        # Normals branch
        self.enc_n_1 = PCBActiv(3, 16, activ='leaky', sample='down-3') # 512 -> 256
        self.enc_n_2 = PCBActiv(16, 64, activ='leaky', sample='down-3') # 256 -> 128; normals out

        self.enc_v_1 = PCBActiv(512, 128, activ='leaky', sample='down-3') # 256 -> 128; skip out
        self.enc_v_2 = PCBActiv(128, 256, activ='leaky', sample='down-5') # 128 -> 64
        self.enc_v_3 = PCBActiv(256, 256, activ='leaky')
        self.enc_v_4 = PCBActiv(256 // 4 + 128 + 64, 512, activ='leaky') # 64 -> 128; skip in; normals in
        self.enc_v_5 = PCBActiv(512 // 4 + 512, 512, bn=True, activ='sigmoid', conv_bias=True)


    def forward(self, visibility, visibility_mask, normals, normals_mask):
        """
        # Arguments
            visibility: float tensor with shape (B, 512, H, W)
            visibility_mask: float tensor with shape (B, 1, H, W)
            normals: float tensor with shape (B, 3, 2 * H, 2 * W)
            normals_mask: float tensor with shape (B, 1, 2 * H, 2 * W)

        # Returns
            visibility_out: float tensor with shape (B, 512, H, W)
            mask_out: float tensor with shape (B, 1, H, W)
        """

        # Training data augmentation
        if self.training and self.data_augment:
            visibility_mask = F.dropout(visibility_mask, p=0.1).gt(0.5).float()

            vnoise = torch.zeros_like(visibility).normal_(0, 0.5)
            vnoise_mask = F.dropout(visibility_mask, p=0.99).gt(0.5).float()
            visibility = torch.clamp(visibility + vnoise_mask * vnoise, 0, 1)

        # Discard inputs with unlikely values, i.e., more than 50% of light and less than 20% on average
        vmean = visibility.mean(dim=1, keepdim=True)
        visibility_mask *= vmean.lt(0.5).float() * vmean.gt(0.2)

        visibility_mask = visibility_mask.tile(1, visibility.size(1), 1, 1)
        normals_mask = normals_mask.tile(1, normals.size(1), 1, 1)

        n, nm = self.enc_n_1(normals, normals_mask)
        n, nm = self.enc_n_2(n, nm)

        v_skip, vm_skip = self.enc_v_1(visibility, visibility_mask)
        v, vm = self.enc_v_2(v_skip, vm_skip)
        v, vm = self.enc_v_3(v, vm)
        # upscale; skip in; normals in
        v = upscale_by_reshape_2d(v, 2)
        vm = upscale_by_reshape_2d(vm, 2)
        v = torch.cat([v, v_skip, n], dim=1)
        vm = torch.cat([vm, vm_skip, nm], dim=1)
        v, vm = self.enc_v_4(v, vm)
        # upscale; input in
        v = upscale_by_reshape_2d(v, 2)
        vm = upscale_by_reshape_2d(vm, 2)
        v = torch.cat([v, visibility], dim=1)
        vm = torch.cat([vm, visibility_mask], dim=1)
        visibility_out, mask_out = self.enc_v_5(v, vm)

        return visibility_out, mask_out[:, :1]


class AlbedoNet(BaseNeuralNet):
    def __init__(self):
        super().__init__()

        # Normals branch
        self.enc_n_1 = PCBActiv(3, 16, sample='down-3')
        self.enc_n_2 = PCBActiv(16, 64)

        self.enc_v_1 = PCBActiv(512, 256)
        self.enc_v_2 = PCBActiv(256, 256)

        self.enc_rgb_1 = PCBActiv(3, 32, sample='down-3')
        self.enc_rgb_2 = PCBActiv(32, 64)

        self.dec_1 = PCBActiv(64 + 256 + 64, 256)
        self.dec_2 = PCBActiv(256, 256)
        self.dec_3 = PCBActiv(256, 256)
        self.dec_4 = PCBActiv(64, 3, bn=True, activ='sigmoid', conv_bias=True)


    def forward(self, tex, tex_mask, visibility, visibility_mask, normals, normals_mask):
        """
        # Arguments
            tex: float tensor with shape (B, 3, H, W)
            tex_mask: float tensor with shape (B, 1, H, W)
            visibility: float tensor with shape (B, 512, H/2, W/2)
            visibility_mask: float tensor with shape (B, 1, H/2, W/2)
            normals: float tensor with shape (B, 3, H, W)
            normals_mask: float tensor with shape (B, 1, H, W)

        # Returns
            albedo: float tensor with shape (B, 3, H, W)
        """
        visibility_mask = visibility_mask.tile(1, visibility.size(1), 1, 1)
        normals_mask = normals_mask.tile(1, normals.size(1), 1, 1)
        tex_mask = tex_mask.tile(1, tex.size(1), 1, 1)

        n, nm = self.enc_n_1(normals, normals_mask)
        n, nm = self.enc_n_2(n, nm)

        v, vm = self.enc_v_1(visibility, visibility_mask)
        v, vm = self.enc_v_2(v, vm)

        c, cm = self.enc_rgb_1(tex, tex_mask)
        c, cm = self.enc_rgb_2(c, cm)

        x = torch.cat([n, v, c], dim=1)
        xm = torch.cat([nm, vm, cm], dim=1)
        x, xm = self.dec_1(x, xm)
        x, xm = self.dec_2(x, xm)
        x, xm = self.dec_3(x, xm)
        x = upscale_by_reshape_2d(x, 2)
        xm = upscale_by_reshape_2d(xm, 2)
        x, xm = self.dec_4(x, xm)

        albedo = 0.77 * x + 0.03
        mask = xm[:, :1]

        return albedo, mask
