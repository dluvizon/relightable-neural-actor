import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

from .modules import Sobel
from .modules import SobelLaplaceOfGaussian
from .geometry import normalize as normalize_fn


def masked_huber_loss(pred, ref, mask, scaling=0.1):
    """ Smooth L1 (huber) loss between the rendered silhouettes and colors.

    # Arguments
        pred, ref: tensors with shape (..., 3)
        mask: tensor with shape (...,)
    """
    mask = mask.unsqueeze(-1)
    _pred = mask * pred + (1 - mask) * ref

    diff = (_pred - ref) ** 2
    loss = ((1 + diff / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)

    loss = loss.abs().sum() / torch.clamp(mask.sum(), 1, None)

    return loss

def masked_L1_loss(pred, ref, mask):
    """ L1 loss between the rendered silhouettes and colors.

    # Arguments
        pred, ref: tensors with shape (..., 3)
        mask: tensor with shape (...,)
    """
    mask = mask.unsqueeze(-1)
    _pred = mask * pred + (1 - mask) * ref

    diff = _pred - ref
    loss = diff.abs().sum() / torch.clamp(mask.sum(), 1, None)

    return loss


class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.
    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range [0,1].
    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at layer :attr:`layer` (default 8, or
    ``relu2_2``) of the pretrained model specified by :attr:`model` (either
    ``'vgg16'`` (default) or ``'vgg19'``).
    If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
    pixels in both height and width will be applied to all images in the input
    and target. The shift will only be applied when the loss function is in
    training mode, and will not be applied if a precomputed feature map is
    supplied as the target.
    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.
    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.
    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.

    From: https://github.com/crowsonkb/vgg_loss/blob/master/vgg_loss.py
    """

    models = {
        'vgg16': torchvision.models.vgg16,
        'vgg19': torchvision.models.vgg19
        }

    def __init__(self, model='vgg16', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = self.models[model](pretrained=True).features[:layer+1]
        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, input):
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = torchvision.transforms.RandomCrop(batch.shape[2:])(padded)

            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]

        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)


def multiscale_gradient_reg(x, mask=None, num_scales=1):
    """Compute a multiscale gradient loss from the input tensor (B, ch, H, W)
    and the (optional) mask tensor (B, 1, H, W)
    """
    if mask is not None:
        mask = mask.gt(0.5).float()
    h, w = x.shape[2:]
    loss = None
    for i in range(num_scales):
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        if mask is not None:
            dh *= mask[:, :, 1:, :] * mask[:, :, :-1, :]
            dw *= mask[:, :, :, 1:] * mask[:, :, :, :-1]
            mask = mask[:, :, ::2, ::2]

        loss_dh = torch.sum(dh) / (w * h)
        loss_dw = torch.sum(dw) / (w * h)
        loss = loss + loss_dh + loss_dw if loss is not None else loss_dh + loss_dw
        x = x[:, :, ::2, ::2]

    return loss


class NeuraCriterium(nn.Module):
    def __init__(self, args, bg_color=1, eps=1e-5):
        super().__init__()
        self.vgg_loss_fn = VGGLoss()
        self.l2_loss_fn = nn.MSELoss()

        self.sobel_chrominance = Sobel(in_channels=2)
        self.bg_color = bg_color
        self.eps = eps

        self.relight = args.relight
        self.coef_loss_l2 = args.coef_loss_l2
        self.coef_loss_vgg = args.coef_loss_vgg
        self.coef_loss_mask = args.coef_loss_mask
        self.coef_loss_hard = args.coef_loss_hard
        self.coef_loss_residual = args.coef_loss_residual
        self.coef_loss_uv_residual = args.coef_loss_uv_residual
        self.coef_loss_albedo = args.coef_loss_albedo
        self.coef_loss_rough = args.coef_loss_rough
        self.coef_uv_normals = args.coef_uv_normals
        self.coef_loss_uv_residual = args.coef_loss_uv_residual
        self.use_uv_nets = args.use_uv_nets
        self.use_uv_nets_norm_only = args.use_uv_nets_norm_only


    def forward(self, data, pred, static_albedo=None, static_brdf=None, uv_mask=None):
        """
        # Arguments
            from `data`:
                'rgb': target rgb color (B=1, C, H, W, 3)
                'mask': target rgb color (B=1, C, H, W)

            from `pred`:
                'rgb': predicted rgb color (C, H, W, 3)
                'prob_hit': predicted probability (C, H, W)
                'uv_residual': predicted uv deformation (C, H, W, 2)
                'sigma': tensor with shape (C, H, W, S=steps+1)
                'probs': tensor with shape (C, H, W, S)
                'residual': residual in canonical space (P, 3)
        """
        meta = {}

        # Get supervision signals
        matting = data['mask'].permute(1, 0, 2, 3) # (C, 1, H, W)
        rgb_target = data['rgb'].squeeze(0).permute(0, 3, 1, 2) # (C, 3, H, W)

        # Get predictions
        prob_hit = pred['prob_hit'].unsqueeze(1) # (C, 1, H, W)
        rgb_pred = pred['rgb'].permute(0, 3, 1, 2) # (C, 3, H, W)
        sigma = pred['sigma'].permute(0, 3, 1, 2) # (B, S, H, W)
        probs = pred['probs'] # (B, S, H, W)

        # L2 loss on RGB images using binary masking
        rgb_l2_pred = rgb_pred + (1 - prob_hit) * self.bg_color
        solid_mask = matting.gt(0.5).float()
        rgb_l2_ref = solid_mask * rgb_target + (1 - solid_mask) * self.bg_color
        meta['loss/rgb_l2'] = self.l2_loss_fn(rgb_l2_pred, rgb_l2_ref)
        meta['image/color_l2'] = rgb_l2_pred
        meta['image/target_l2'] = rgb_l2_ref

        # VGG loss on RGB images using matting
        rgb_vgg_pred = matting * rgb_pred + (1 - matting) * rgb_target
        meta['loss/rgb_vgg'] = self.vgg_loss_fn(rgb_vgg_pred, rgb_target)
        meta['image/color_vgg'] = rgb_vgg_pred
        meta['image/target_vgg'] = rgb_target

        # Mask (sigma) loss
        sigma = torch.sigmoid(sigma)
        meta['loss/mask'] = ((1 - solid_mask) * sigma).mean()
        meta['image/prob_hit'] = prob_hit
        meta['image/target_mask'] = matting

        # Hard loss LOLNeRF
        beta_distrib = torch.exp(-torch.abs(probs)) + torch.exp(-torch.abs(1-probs))
        meta['loss/hard'] = torch.mean(-torch.log(beta_distrib))

        # Aggregate general losses
        loss = (
            self.coef_loss_l2 * meta['loss/rgb_l2']
            + self.coef_loss_vgg * meta['loss/rgb_vgg']
            + self.coef_loss_mask * meta['loss/mask']
            + self.coef_loss_hard * meta['loss/hard']
        )

        # Residual regularization (in 3D canonical space)
        if ('residual' in pred) and (pred['residual'] is not None):
            meta['loss/residual3d'] = (pred['residual'] ** 2).mean()
            loss += self.coef_loss_residual * meta['loss/residual3d']

        if self.relight:
            relight_pred = pred['relight'].permute(0, 3, 1, 2) # (C, 3, H, W)
            albedo = pred['albedo'].permute(0, 3, 1, 2) # (C, 3, H, W)
            rough = pred['rough'].permute(0, 3, 1, 2) # (C, 1, H, W)
            visibility = pred['sampled_visibility'].permute(0, 3, 1, 2) # (C, 512, H, W)
            # lights = visibility.mean(dim=1, keepdims=True).detach().clone() # (C, 1, H, W)

            # L2 loss on relight results
            relight_l2_pred = relight_pred + (1 - prob_hit) * self.bg_color
            meta['loss/relight_l2'] = self.l2_loss_fn(relight_l2_pred, rgb_l2_ref)
            meta['image/relight_l2'] = relight_l2_pred

            # VGG loss on relight results
            relight_vgg_pred = matting * relight_pred + (1 - matting) * rgb_target
            meta['loss/relight_vgg'] = self.vgg_loss_fn(relight_vgg_pred, rgb_target)
            meta['image/relight_vgg'] = relight_vgg_pred

            if static_albedo is not None:
                meta['loss/static_albedo'] = multiscale_gradient_reg(static_albedo, num_scales=3)
                meta['image/static_albedo'] = static_albedo.flip(2)
            else:
                meta['loss/static_albedo'] = 0 * meta['loss/rgb_l2']
            if static_brdf is not None:
                meta['loss/static_brdf'] = multiscale_gradient_reg(static_brdf, num_scales=3)
                meta['image/static_brdf'] = static_brdf.flip(2)
            else:
                meta['loss/static_brdf'] = 0 * meta['loss/rgb_l2']

            # Albedo and BRDF regularization in camera space
            Y, CbCr = convert_rgb_to_ycbcr(rgb_target)
            chrom_edges = self.sobel_chrominance(CbCr)
            edge_mask = solid_mask * chrom_edges.lt(0.01).float()

            meta['image/chrom_edges'] = chrom_edges / torch.clamp(chrom_edges.max(), 1e-5, None)
            meta['image/edge_mask'] = edge_mask

            meta['loss/albedo'] = multiscale_gradient_reg(albedo, edge_mask)
            meta['loss/rough'] = multiscale_gradient_reg(rough, edge_mask)
            meta['image/albedo'] = albedo
            meta['image/rough'] = rough

            # UV residual regularization
            if (pred['uv_residual'] is not None) and (pred['sampled_uv_residual'] is not None):
                uv_residual = pred['uv_residual'].permute(0, 3, 1, 2) # (C, 3, H, W)
                sampled_uv_residual = pred['sampled_uv_residual'] # (P, S, 2)

                loss_uv_res_l2 = (sampled_uv_residual ** 2).mean()
                loss_uv_res_grad = multiscale_gradient_reg(uv_residual)
                meta['loss/uv_residual'] = loss_uv_res_l2 + loss_uv_res_grad
                uv_div = torch.clamp(uv_residual.max(), 1e-6, None)
                u = uv_residual[:, 0:1] / uv_div #/ torch.clamp(uv_residual[:, 0:1].max(), 1e-5, None)
                v = uv_residual[:, 1:2] / uv_div #/ torch.clamp(uv_residual[:, 1:2].max(), 1e-5, None)
                meta['image/uv_residual'] = torch.cat([u, v, u ** 2 + v ** 2], dim=1)
            else:
                meta['loss/uv_residual'] = 0 * meta['loss/rgb_l2']

            # Aggregate relight losses
            loss += (
                self.coef_loss_l2 * meta['loss/relight_l2']
                + self.coef_loss_vgg * meta['loss/relight_vgg']
                + self.coef_loss_albedo * meta['loss/static_albedo']
                + self.coef_loss_rough * meta['loss/static_brdf']
                + self.coef_loss_albedo * meta['loss/albedo']
                + self.coef_loss_rough * meta['loss/rough']
                + self.coef_loss_uv_residual * meta['loss/uv_residual']
            )

            # If using UV Nets, supervised NormalNet and VisibilityNet
            if self.use_uv_nets:
                normals_target = data['uv_normals_dense'].permute(0, 3, 1, 2) # (B, 3, H, W)
                normals_target_mask = data['uv_normals_dense_mask'].unsqueeze(1) # (B, 1, H, W)
                normals_pred = pred['uv_normals_pred']

                meta['loss/uv_normal'] = torch.mean(normals_target_mask * (normals_target - normals_pred) ** 2)
                # meta['loss/uv_normal'] = torch.sum(normals_target_mask * torch.abs(normals_target - normals_pred)) / torch.clamp(torch.sum(normals_target_mask), 1, None)
                meta['image/normals_target'] = (normals_target + 1) / 2
                normals_pred_n = normalize_fn(torch.clamp(normals_pred, -1, 1), axis=1)[0]
                meta['image/normals_pred'] = (normals_pred_n + 1) / 2
                loss += self.coef_uv_normals * meta['loss/uv_normal']

                if not self.use_uv_nets_norm_only:
                    vis_target = data['uv_vis_dense'].permute(0, 3, 1, 2) # (B, 512, H, W)
                    vis_target_mask = data['uv_vis_dense_mask'].unsqueeze(1) # (B, 1, H, W)
                    uv_vis_pred = pred['uv_vis_pred']

                    x = torch.clamp(uv_vis_pred, self.eps, 1 - self.eps)
                    y = torch.clamp(vis_target, 0, 1)
                    bc = -((y * torch.log(x) + (1 - y) * torch.log(1 - x)))
                    meta['loss/uv_visibility'] = torch.sum(vis_target_mask * bc) / torch.clamp(torch.sum(vis_target_mask), 1, None)
                    
                    meta['image/vis_target'] = 2 * vis_target.mean(dim=1)
                    meta['image/vis_pred'] = 2 * uv_vis_pred.mean(dim=1)
                    loss += meta['loss/uv_visibility']

        return loss, meta


class NeuraNormalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_loss_fn = VGGLoss()

    def forward(self, pred_normals, normals, mask):
        """
        # Arguments
        pred_normals: tensors with shape (B, 3, H, W)
        normals: tensors with shape (B, 3, H, W)
        mask: tensors with shape (B, 1, H, W)
        """
        pred_normals = mask * pred_normals
        normals = mask * normals

        diff = pred_normals - normals
        loss_l2 = torch.mean(diff ** 2)
        loss_vgg = self.vgg_loss_fn(pred_normals * 0.5 + 0.5, normals * 0.5 + 0.5)

        return loss_l2, loss_vgg


class NeuraVisibilityLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, pred_visibility, visibility, mask):
        """
        # Arguments
        pred_visibility: tensors with shape (B, L, H, W)
        visibility: tensors with shape (B, L, H, W)
        mask: tensors with shape (B, 1, H, W)
        """
        x = torch.clamp(pred_visibility, self.eps, 1 - self.eps)
        y = torch.clamp(visibility, 0, 1)
        loss_bc = -((y * torch.log(x) + (1 - y) * torch.log(1 - x)))
        loss_bc = torch.mean(mask * loss_bc)

        return loss_bc


def convert_rgb_to_ycbcr(x):
    """Convert the RGb input to luminance and chrominance.

    # Arguments
        x: RGB shape (B, 3, H, W)

    # Returns
        Y: shape (B, 1, H, W)
        CbCr: shape (B, 2, H, W)
    """
    Y = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
    Cb = -0.1146 * x[:, 0:1] - 0.3854 * x[:, 1:2] + 0.5 * x[:, 2:3]
    Cr = 0.5 * x[:, 0:1] - 0.4542 * x[:, 1:2] - 0.0458 * x[:, 2:3]

    return Y, torch.cat([Cb, Cr], dim=1)


class NeuraTextureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.vgg_loss_fn = VGGLoss()
        self.sobel_chrominance = Sobel(in_channels=2)
        self.sobel_rgb = Sobel(in_channels=3)

    def forward(self, pred_texture, albedo, texture, mask):
        """
        # Arguments
        pred_texture: tensors with shape (B, 3, H, W)
        texture: tensors with shape (B, 3, H, W)
        mask: tensors with shape (B, 1, H, W)
        """
        x = mask * pred_texture
        y = mask * texture

        loss_l2 = torch.mean((x - y) ** 2)
        # loss_vgg = self.vgg_loss_fn(pred_texture, texture)

        tex_Y, tex_CbCr = convert_rgb_to_ycbcr(texture)
        chrom_edges = mask * self.sobel_chrominance(tex_CbCr)
        edge_mask = chrom_edges.lt(0.002).float()
        albedo_edges = self.sobel_rgb(albedo)
        loss_reg = torch.mean(edge_mask * albedo_edges)

        return loss_l2, loss_reg, {
            'chrom_edges': chrom_edges / torch.clamp(chrom_edges.max(), 1e-4, None),
            'albedo_edges': albedo_edges,
            'edge_mask': edge_mask,
        }

