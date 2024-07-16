import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import torchvision

import copy

from .geometry import normalize


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class NeRFPosEmbLinear(nn.Module):

    def __init__(self, in_dim, out_dim, angular=False, no_linear=False, cat_input=False, no_pi=False):
        super().__init__()
        assert out_dim % (2 * in_dim) == 0, "dimension must be dividable"
        L = out_dim // 2 // in_dim
        emb = torch.exp(torch.arange(L, dtype=torch.float) * math.log(2.))
        
        if (not angular) and (not no_pi):
            emb = emb * math.pi
            
        self.emb = nn.Parameter(emb, requires_grad=False)
        self.angular = angular
        self.linear = Linear(out_dim, out_dim) if not no_linear else None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cat_input = cat_input

    def forward(self, x):
        assert x.size(-1) == self.in_dim, "size must match"
        sizes = x.size() 
        inputs = x.clone()

        if self.angular:
            x = torch.acos(x.clamp(-1 + 1e-6, 1 - 1e-6))
        x = x.unsqueeze(-1) @ self.emb.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(*sizes[:-1], self.out_dim)
        if self.linear is not None:
            x = self.linear(x)
        if self.cat_input:
            x = torch.cat([x, inputs], -1)
        return x

    def extra_repr(self) -> str:
        outstr = 'Sinusoidal (in={}, out={}, angular={})'.format(
            self.in_dim, self.out_dim, self.angular)
        if self.cat_input:
            outstr = 'Cat({}, {})'.format(outstr, self.in_dim)
        return outstr


class FCLayer(nn.Module):
    """
    Reference:
        https://github.com/vsitzmann/pytorch_prototyping/blob/10f49b1e7df38a58fd78451eac91d7ac1a21df64/pytorch_prototyping.py
    """
    def __init__(self, in_dim, out_dim, with_ln=True, use_softplus=False, non_linear=True):
        super().__init__()
        self.net = [nn.Linear(in_dim, out_dim)]
        if with_ln:
            self.net += [nn.LayerNorm([out_dim])]
        if non_linear:
            self.net += [nn.ReLU()] if not use_softplus else [nn.Softplus(beta=100)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x) 


class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False,
                 with_ln=True,
                 use_softplus=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features, hidden_ch, with_ln, use_softplus))
        for i in range(num_hidden_layers):
            self.net.append(FCLayer(hidden_ch, hidden_ch, with_ln, use_softplus))
        if outermost_linear:
            self.net.append(Linear(hidden_ch, out_features))
        else:
            self.net.append(FCLayer(hidden_ch, out_features, with_ln, use_softplus))
        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)

    def __getitem__(self, item):
        return self.net[item]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, input):
        return self.net(input)


class ImplicitField(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, 
                outmost_linear=False, with_ln=True, skips=None, 
                spec_init=True, use_softplus=False, experts=1):
        super().__init__()
        self.skips = skips
        self.net = []
        self.total_experts = experts
        prev_dim = in_dim
        for i in range(num_layers):
            next_dim = out_dim if i == (num_layers - 1) else hidden_dim
            if (i == (num_layers - 1)) and outmost_linear:
                if experts <= 1:
                    module = nn.Linear(prev_dim, next_dim)
                else:
                    raise NotImplementedError
            else:
                if experts <= 1:
                    module = FCLayer(prev_dim, next_dim, with_ln=with_ln, use_softplus=use_softplus)
                else:
                    raise NotImplementedError

            self.net.append(module)
                
            prev_dim = next_dim
            if (self.skips is not None) and (i in self.skips) and (i != (num_layers - 1)):
                prev_dim += in_dim
        
        if num_layers > 0:
            self.net = nn.ModuleList(self.net)
            if spec_init:
                self.net.apply(self.init_weights)


    def forward(self, x, sizes=None):
        """
        # Arguments
            x: tensor shape (num_rays, in_dim)

        # Returns
            Output tensor with shape (num_rays, out_dim)
        """
        y = self.net[0]([x, sizes] if self.total_experts > 1 else x)
        for i in range(len(self.net) - 1):
            if (self.skips is not None) and (i in self.skips):
                y = torch.cat((x, y), dim=-1) / math.sqrt(2)    # BUG: I found IDR has sqrt(2)
            y = self.net[i+1]([y, sizes] if self.total_experts > 1 else y)
        return y

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


class TextureField(ImplicitField):
    """
    Pixel generator based on 1x1 conv networks
    """
    def __init__(self, in_dim, hidden_dim, num_layers, 
                with_alpha=False, with_ln=True, monochromatic=False,
                spec_init=True, experts=0):

        if monochromatic:
            out_dim = 1
        else:
            out_dim = 3 if not with_alpha else 4

        super().__init__(in_dim, out_dim, hidden_dim, num_layers, 
            outmost_linear=True, with_ln=with_ln, spec_init=spec_init, experts=experts)


class TriplaneEncoder(nn.Module):
    def __init__(self, num_channels, resolution, aggregation='sum'):
        super().__init__()
        assert aggregation in ['sum', 'concat'], (f'Invalid aggregation mode ({aggregation})!')
        self.aggregation = aggregation
        self.num_channels = num_channels

        tnames = ['T_xy', 'T_yz', 'T_zx']
        for n in tnames:
            x = torch.empty((1, num_channels, resolution, resolution), requires_grad=True)
            x.data.normal_(0, 0.1)
            setattr(self, n, nn.Parameter(x))


    def forward(self, xyz):
        """
        # Inputs:
            x: (N, 3)

        # Outputs:
            Aggregated features with shape (N, f_dim), where
                f_dim = num_channels if aggregation == 'sum'
                f_dim = 3 * num_channels if aggregation == 'concat'
        """
        grids_xy = xyz[None, :, None, [0, 1]]
        grids_yz = xyz[None, :, None, [1, 2]]
        grids_zx = xyz[None, :, None, [2, 0]]
        f_xy = F.grid_sample(self.T_xy, grids_xy, mode='bilinear', align_corners=False)
        f_yz = F.grid_sample(self.T_yz, grids_yz, mode='bilinear', align_corners=False)
        f_zx = F.grid_sample(self.T_zx, grids_zx, mode='bilinear', align_corners=False)

        if self.aggregation == 'sum':
            feat = f_xy + f_yz + f_zx
        elif self.aggregation == 'concat':
            feat = torch.concat([f_xy, f_yz, f_zx], dim=1)

        return feat.permute(0, 2, 3, 1).reshape(-1, self.num_channels)


class SignedDistanceField(ImplicitField):
    """
    Predictor for density or SDF values.
    """
    def __init__(self, in_dim, hidden_dim, num_layers=1, 
                with_ln=True, spec_init=True):
        super().__init__(
            in_dim, in_dim, in_dim, num_layers - 1,
            with_ln=with_ln, spec_init=spec_init)

        if num_layers > 0:
            self.hidden_layer = FCLayer(in_dim, hidden_dim, with_ln)
        else:
            self.hidden_layer = None

        prev_dim = hidden_dim if num_layers > 0 else in_dim
        self.output_layer = nn.Linear(prev_dim, 1)
        self.output_layer.bias.data.fill_(0.5)   # set a bias for density
        
    def forward(self, x):
        if self.hidden_layer is not None:
            return self.output_layer(self.hidden_layer(x)).squeeze(-1), None
        return self.output_layer(x).squeeze(-1), None


class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """
    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )

        print("Using torchvision", backbone, "encoder")
        self.model = getattr(torchvision.models, backbone)(
            pretrained=pretrained, norm_layer=norm_layer
        )
        # Following 2 lines need to be uncommented for older configs
        self.model.fc = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                # recompute_scale_factor=True,
            )
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        latents = [x]
        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.model.maxpool(x)
            x = self.model.layer1(x)
            latents.append(x)
        if self.num_layers > 2:
            x = self.model.layer2(x)
            latents.append(x)
        if self.num_layers > 3:
            x = self.model.layer3(x)
            latents.append(x)
        if self.num_layers > 4:
            x = self.model.layer4(x)
            latents.append(x)

        self.latents = latents
        align_corners = None if self.index_interp == "nearest " else True
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode=self.upsample_interp,
                align_corners=align_corners,
            )
        
        return torch.cat(latents, dim=1)

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )


class NeuRAFeatureDecoder(nn.Module):
    def __init__(self, input_dim, texture_feature_dim=32):
        super().__init__()

        self.input_gn = nn.GroupNorm(4, input_dim)
        self.input_relu = nn.ReLU()

        self.delta_conv = nn.Conv2d(input_dim, 3, 3, stride=1, padding=1)
        self.delta_conv.weight.data *= 1e-5
        self.delta_conv.bias.data *= 0

        self.conv_tex = nn.Conv2d(input_dim // 4, texture_feature_dim, 3, stride=1, padding=1)
    
    def forward(self, x):
        """
        :param x image (B, C, H, W)
        :return
            deformation_xyz (B, 3, H, W)
            tex_feature_map (B, 32, 2*H, 2*W)
        """
        x = self.input_relu(self.input_gn(x))
        deformation_xyz = self.delta_conv(x)

        batch, ch, h, w = x.shape
        x = torch.reshape(x, (batch, ch // 4, 2*h, 2*w))
        tex_feature_map = self.conv_tex(x)
        
        return deformation_xyz, tex_feature_map


class NeuraSpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """
    def __init__(self, use_normals=True):
        super().__init__()

        if use_normals:
            input_dim = 6
        else:
            input_dim = 3

        self.enc_conv1 = nn.Conv2d(input_dim, 32, 7, stride=4, padding=2) # 512 -> 128

        self.enc_gn2 = nn.GroupNorm(4, 32)
        self.enc_relu2 = nn.ReLU()
        self.enc_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=2) # 128 -> 64
    
        self.enc_gn2 = nn.GroupNorm(4, 64)
        self.enc_relu2 = nn.ReLU()

        self.enc_conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=0) # 64 -> 32
        self.enc_gn3 = nn.GroupNorm(4, 128)
        self.enc_relu3 = nn.ReLU()

        self.enc_conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # 32 -> 16
        self.enc_gn4 = nn.GroupNorm(4, 256)
        self.enc_relu4 = nn.ReLU()

        self.enc_conv5 = nn.Conv2d(256, 256, 3, stride=2, padding=1) # 16 -> 8
        self.enc_gn5 = nn.GroupNorm(4, 256)
        self.enc_relu5 = nn.ReLU()

        self.enc_conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.enc_gn6 = nn.GroupNorm(4, 256)
        self.enc_relu6 = nn.ReLU()

        self.enc_conv7 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.enc_gn7 = nn.GroupNorm(4, 256)
        self.enc_relu7 = nn.ReLU()

        # TODO: how to arrange skip connections?

        self.dec_conv5 = nn.ConvTranspose2d(256+256, 256, 4, stride=2, padding=2) # 8 -> 16
        self.dec_gn5 = nn.GroupNorm(4, 256)
        self.dec_relu5 = nn.ReLU()

        self.dec_conv4 = nn.ConvTranspose2d(256+128, 192, 4, stride=2, padding=2) # 16 -> 32
        self.dec_gn4 = nn.GroupNorm(4, 192)
        self.dec_relu4 = nn.ReLU()

        self.dec_conv3 = nn.ConvTranspose2d(192+128, 160, 4, stride=2, padding=2) # 32 -> 64
        self.dec_gn3 = nn.GroupNorm(4, 160)
        self.dec_relu3 = nn.ReLU()

        self.dec_conv2 = nn.ConvTranspose2d(160+64, 128, 4, stride=2, padding=2) # 64 -> 128
        self.dec_gn2 = nn.GroupNorm(4, 128)
        self.dec_relu2 = nn.ReLU()

        self.dec_relu1 = nn.ReLU()
        self.dec_conv1 = nn.ConvTranspose2d(128+32, 96, 4, stride=2, padding=2) # 128 -> 256
        self.dec_gn1 = nn.GroupNorm(4, 96)



    def forward(self, x):
        """
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H/2, W/2)
        """
        x = self.enc_conv1(x)
        skip1 = self.enc_gn1(x)
        x = self.enc_relu1(skip1)
        
        return x

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )


class BaseNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

    def summary(self):
        print (self)
        num_par = sum(p.numel() for p in self.parameters() if p.requires_grad)
        units = ['', 'K', 'M', 'G']
        for i in range(len(units)):
            if num_par > 1e3:
                num_par /= 1e3
            else:
                break
        if hasattr(self, 'name'):
            name = self.name
        else:
            name = 'BaseNeuralNet'

        print (f"Total parameters ({name}): {num_par:.1f} {units[i]}")


class AddCoordEncoding2D(nn.Module):
    def __init__(self, input_dim, num_rows, num_cols, num_frequencies=4):
        super().__init__()
        xx_range = torch.arange(num_rows, dtype=torch.int32)
        yy_range = torch.arange(num_cols, dtype=torch.int32)
        xx_range = np.pi * (xx_range[None, None, None, :].float() / (num_rows - 1))
        yy_range = np.pi * (yy_range[None, None, :, None].float() / (num_cols - 1))
        encoding = []
        for i in range(num_frequencies):
            for fn in [torch.sin, torch.cos]:
                encoding.append(fn((2 ** i) * xx_range).repeat(1, 1, num_cols, 1))
                encoding.append(fn((2 ** i) * yy_range).repeat(1, 1, 1, num_rows))
        encoding = torch.cat(encoding, dim=1)

        self.conv = nn.Conv2d(encoding.shape[1], input_dim, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.encoding = encoding

    def forward(self, input_tensor):
        if input_tensor.is_cuda and (self.encoding.is_cuda == False):
            self.encoding = self.encoding.cuda()

        x = self.conv(self.encoding)
        x = self.relu(x)
        output = x + input_tensor

        return output


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)

        return x * y.expand_as(x)


class Residual(nn.Module):
    def __init__(self, input_dim, mid_dim, output_dim,
            add_coord=False,
            num_rows=None,
            num_cols=None,
            num_frequencies=None,
            use_group_norm=True):
        super().__init__()

        if use_group_norm:
            norm_fn = lambda x: nn.GroupNorm(4, x)
        else:
            norm_fn = nn.BatchNorm2d

        self.bn1 = norm_fn(input_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_dim, mid_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_fn(mid_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.se = SE_Block(output_dim)

        if input_dim == output_dim:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0)

        self.add_coord = add_coord
        if self.add_coord:
            assert (isinstance(num_rows, int)
                and isinstance(num_cols, int)
                and isinstance(num_frequencies, int)), (
                    f"When using 'add_coord', additional parameters should be provided!"
                )
            self.coord_enc = AddCoordEncoding2D(mid_dim, num_rows, num_cols, num_frequencies)


    def forward(self, x):
        residual = self.skip(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        if self.add_coord:
            x = self.coord_enc(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.se(x)
        return x + residual


class UpScaleBlock(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = nn.functional.interpolate(x,
                scale_factor=self.scale_factor,
                mode=self.mode,
                align_corners=self.align_corners)
        return x


class CoordUNet(BaseNeuralNet):
    def __init__(self, num_levels, params,
                 input_dim=3,
                 top_level=True,
                 add_coord=False,
                 num_rows=None,
                 num_cols=None):
        super().__init__()
        assert num_levels == len(params), (
            f'`params` ({params}) should contain `num_levels` ({num_levels}) entries!'
        )

        params = copy.deepcopy(params)
        args = params.pop(0)

        self.top_level = top_level
        if self.top_level:
            self.first_layer = nn.Conv2d(input_dim, args['input_dim'], kernel_size=3, stride=2, padding=1)

        self.up1 = Residual(**args)
        args['input_dim'] = args['output_dim']

        self.pool = nn.MaxPool2d(2, stride=2)
        self.low1 = Residual(**args)

        if num_levels > 1:
            self.low2 = CoordUNet(num_levels - 1, params, top_level=False,
                    add_coord=add_coord,
                    num_rows=num_rows // 2 if add_coord is not None else None,
                    num_cols=num_cols // 2 if add_coord is not None else None)
            args['input_dim'] = params[0]['output_dim']
        else:
            self.low2 = Residual(**args)
            args['input_dim'] = args['output_dim']

        self.up2 = UpScaleBlock()
        self.up3 = Residual(**args)

        if self.top_level:
            args['input_dim'] = args['output_dim']
            self.last_layer = Residual(**args)

    # Defining the forward pass    
    def forward(self, x):
        if self.top_level:
            x = self.first_layer(x)
        x = self.up1(x)
        identity = x
        x = self.pool(x)
        x = self.low1(x)
        x = self.low2(x)
        x = self.up2(x)
        x = self.up3(x)
        x = x + identity
        if self.top_level:
            x = self.last_layer(x)

        return x


class TextureEncoder(BaseNeuralNet):
    def __init__(self, add_coord=True, num_rows=None, num_cols=None, use_normals=True):
        self.name = self.__class__.__name__
        super().__init__()

        if use_normals:
            input_dim = 6
        else:
            input_dim = 3

        nof = 32
        model_params = [
            {'input_dim':  32, 'mid_dim':  24, 'output_dim': nof, 'num_frequencies': 4},
            {'input_dim': nof, 'mid_dim':  32, 'output_dim':  64, 'num_frequencies': 6},
            {'input_dim':  64, 'mid_dim':  48, 'output_dim': 128, 'num_frequencies': 2},
            {'input_dim': 128, 'mid_dim':  80, 'output_dim': 144, 'num_frequencies': 2},
            {'input_dim': 144, 'mid_dim': 128, 'output_dim': 192, 'num_frequencies': 1},
            {'input_dim': 192, 'mid_dim': 128, 'output_dim': 256, 'num_frequencies': 1},
        ]
        self.latent_size = nof
        self.coord_unet = CoordUNet(len(model_params), model_params,
                                    input_dim=input_dim,
                                    add_coord=add_coord,
                                    num_rows=num_rows,
                                    num_cols=num_cols)
        # self.output_layer = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(nof, 3, kernel_size=3, stride=1, padding=1),
        # )

    def forward(self, x):
        x = self.coord_unet(x)
        # x = self.output_layer(x)
        return x



class TextureEncoder2(BaseNeuralNet):
    def __init__(self):
        self.name = self.__class__.__name__
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(4, 32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.latent_size = 64


    def forward(self, x):
        x = torch.sigmoid(x)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        return x


class PoseLatentMap(BaseNeuralNet):
    def __init__(self, num_rows, num_cols, num_channels, num_samples):
        self.name = self.__class__.__name__
        super().__init__()



    def forward(self, x):
        return x



def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class PartialConv(nn.Module):
    """Implementation borrowed from
    https://github.com/naoto0804/pytorch-inpainting-with-partial-conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        #
        # input: shape (B, ch, H, W)
        # mask: shape (B, 1, H, W)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)
            # output_mask = torch.tile(output_mask, (1, output.size(1), 1, 1))

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output_mask)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        elif activ == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


def upscale_by_reshape_2d(input, factor=2):
    """This function takes as input a tensor of shape (B, C, H, W), and
    reshapes the tensor by slicing the feature space into segments that are
    rearranged in the spacial domain, resulting in a tensor of shape (B, C//(s*s), s*H, s*W),
    where s=factor is the rescaling factor.
    For instance:

    The input tensor:               is converted to with s=2:
      tensor([[[ 1.,  5.],            tensor([[[ 1.,  3.,  5.,  0.],
               [ 9., 13.]],                    [ 2.,  4.,  0.,  0.],
              [[ 2.,  0.],                     [ 9.,  0., 13.,  0.],
               [ 0.,  0.]],                    [ 0.,  0.,  0.,  0.]]])
              [[ 3.,  0.],
               [ 0.,  0.]],
              [[ 4.,  0.],
               [ 0.,  0.]]])
    """
    s = factor
    x = input
    batch, ch, h, w = x.shape
    x = x.permute((0, 2, 3, 1)).reshape((batch, h, s*w, ch//s))
    x = x.permute((0, 1, 3, 2)).reshape((batch, s*h, ch//(s*s), s*w))
    return x.permute((0, 2, 1, 3))


class Sobel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1).tile((in_channels, in_channels, 1, 1))
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        # x = torch.abs(x)
        x = torch.sum(x, dim=1, keepdim=True)
        # x = torch.sqrt(x) # This leads to instability, i.e., NaNs
        return x


class SobelLaplaceOfGaussian(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Gaussian filter
        # self.gaussian = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # G = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
        # G = G.unsqueeze(0).unsqueeze(1).tile((in_channels, in_channels, 1, 1))
        # self.gaussian.weight = nn.Parameter(G, requires_grad=False)

        # Sobel and Laplacian filters
        self.sobel_laplace = nn.Conv2d(in_channels=in_channels, out_channels=3 * in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        Sx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Sy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        L = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        W = torch.cat([Sx.unsqueeze(0), Sy.unsqueeze(0), L.unsqueeze(0)], 0)
        W = W.unsqueeze(1).tile((in_channels, in_channels, 1, 1))
        self.sobel_laplace.weight = nn.Parameter(W, requires_grad=False)

    def forward(self, img):
        # x = self.gaussian(img)
        x = img
        x = self.sobel_laplace(x)
        x = torch.abs(x)
        x = torch.sum(x, dim=1, keepdim=True)
        return x


class Depth2Normal(nn.Module):
    """Compute normal map from depth map

    # Arguments
        ray_start: tensor (1, 3)
        ray_dir: tensor (N=H*W, 3)
        depths: regressed depth values from the implicit field (N, 1)
        size: size of the image from where rays were sampled (W, H)

    # Returns
        Normals as a tensor with shape (H, W, 3) and xyz coordinates as (H, W, 3)
    """
    def __init__(self):
        super().__init__()
        self.filter_x = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3,
                                  stride=1, padding=1, padding_mode='replicate', bias=False)
        self.filter_y = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3,
                                  stride=1, padding=1, padding_mode='replicate', bias=False)

        Gx = torch.tensor([[[[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]])
        Gy = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]]]])

        self.filter_x.weight = nn.Parameter(Gx.tile(3, 1, 1, 1), requires_grad=False)
        self.filter_y.weight = nn.Parameter(Gy.tile(3, 1, 1, 1), requires_grad=False)


    def forward(self, ray_start, ray_dir, depths, size):
        W, H = size
        assert H * W == depths.size(0), (f'Error! Invalid size ({size}) for input tensor {depths.shape}')

        coords = ray_start + ray_dir * depths
        coords_wh = coords.view(1, H, W, 3).permute(0, 3, 1, 2) # (1, 3, H, W)

        dx = self.filter_x(coords_wh)
        dy = self.filter_y(coords_wh)
        dx = normalize(dx, axis=1)[0]
        dy = normalize(dy, axis=1)[0]

        normal = torch.cross(dx, dy, dim=1)
        normal = normalize(normal, axis=1)[0]

        return normal[0].permute(1, 2, 0), coords_wh[0].permute(1, 2, 0)
