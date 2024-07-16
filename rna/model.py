import os
import numpy as np
import cv2

import torch
import torch.nn.functional as F

from fairnr.clib import cloud_ray_intersect
from fairnr.clib._ext import point_face_dist_forward

from .geometry import ray
from .geometry import gen_light_ray_dir
from .geometry import cart2sph, sph2cart
from .geometry import normalize as normalize_fn
from .modules import NeRFPosEmbLinear
from .modules import ImplicitField
from .modules import SignedDistanceField
from .modules import TextureField
from .modules import FCBlock
from .modules import SpatialEncoder
from .modules import BaseNeuralNet
from .modules import Depth2Normal
from .cnnmodels import NormalNet
from .cnnmodels import VisibilityNet
from .microfacet import Microfacet
from .utils import masked_scatter, masked_scatter_value
from .utils import func_linear2srgb
from .utils import uv_grid_to_uv_idx
from .utils import func_inverse_sigmoid_np
from .utils import compute_barycentric_coordinates
from .io import save_points_lines_obj
from .io import save_color_pc_obj

MAX_DEPTH = 10.0


class NeuRA(BaseNeuralNet):
    def __init__(self, args, faces, face_uv, skinning_weights,
            tpose_RT=None, uv_mask=None, uv_deform_mask=None, relight=False,
            canonical_mesh_bounds=None) -> None:
        """
        # Arguments
            faces: indexes of the human mesh faces, integer array with shape (F, 3)
            face_uv: UV mapping array with shape (F, 3, 2)
            tpose_RT: (optional) specific T-Pose parameters as an array with shape (J, 4, 4), or None
            uv_mask: (optional) UV map masking as (tex_h, tex_w), or None
            uv_deform_mask: (optional) UV map masking for UV deformation as (tex_h, tex_w), or None.
                If None, assume UV deformation as all ones.
        """
        super().__init__()
        self.min_dis_eps = args.min_dis_eps
        self.max_num_pts_processing = args.max_num_pts_processing
        self.max_num_rays_intersecting = args.max_num_rays_intersecting
        self.envmap_size = args.envmap_size
        self.num_steps = args.num_points_per_ray
        self.canonical_mesh_bounds = canonical_mesh_bounds
        self.clip_canonical = args.clip_canonical
        self.use_uv_nets = args.use_uv_nets
        self.use_uv_nets_norm_only = args.use_uv_nets_norm_only

        self.relight = relight
        self.albedo_scale = args.albedo_scale
        self.albedo_bias = args.albedo_bias
        self.static_albedo = args.static_albedo
        self.coef_envmap_light_rgb_factor = args.coef_envmap_light_rgb_factor
        self.envmap_factor = args.envmap_factor

        self.geometry_layers = [] # list all base layers required for geometry
        self.radiance_layers = [] # list only layers specific to radiance field (direct RGB prediction)
        self.relight_layers = [] # list only layers specific to relight function (albedo, BRDF, ...)

        if canonical_mesh_bounds is not None:
            self.register_buffer("canonical_mesh_bounds_min", torch.from_numpy(np.array(canonical_mesh_bounds['min'], dtype=np.float32)), persistent=False)
            self.register_buffer("canonical_mesh_bounds_max", torch.from_numpy(np.array(canonical_mesh_bounds['max'], dtype=np.float32)), persistent=False)
        else:
            self.canonical_mesh_bounds_min = None
            self.canonical_mesh_bounds_max = None

        self.register_buffer("faces", torch.from_numpy(np.array(faces, dtype=np.int_)), persistent=False)
        self.register_buffer("face_uv", torch.from_numpy(np.array(face_uv, dtype=np.float32)), persistent=False)

        if uv_mask is not None:
            uv_mask = cv2.resize(uv_mask, (512, 512), interpolation=cv2.INTER_AREA)
            uv_mask = uv_mask[None, None, ...] # (1, 1, H, W)
            self.register_buffer("uv_mask", torch.from_numpy(uv_mask).gt(uv_mask.max() / 2).flip(2).float(), persistent=False)

        if uv_deform_mask is not None:
            uv_deform_mask = cv2.resize(uv_deform_mask, (512, 512), interpolation=cv2.INTER_AREA)
            uv_deform_mask = uv_deform_mask[None, None, ...] # (1, 1, H, W)
            self.register_buffer("uv_deform_mask", torch.from_numpy(uv_deform_mask).gt(uv_deform_mask.max() / 2).flip(2).float(), persistent=False)
        else:
            self.register_buffer("uv_deform_mask", torch.ones(size=(1, 1, 512, 512), dtype=torch.float32), persistent=False)


        self.register_buffer("skinning_weights", torch.from_numpy(np.array(skinning_weights, dtype=np.float32)), persistent=False)
        if tpose_RT is not None:
            tpose_RT = tpose_RT.reshape(1, -1, 16) # (B=1, J, 4x4)
            self.register_buffer("tpose_RT", torch.from_numpy(np.array(tpose_RT, dtype=np.float32)), persistent=False)
        else:
            self.tpose_RT = None

        # Define the ray start at the surface point
        env_w, env_h = self.envmap_size
        envmap_xyz, envmap_dir_wld, envmap_area = gen_light_ray_dir(env_h, env_w)
        self.register_buffer("envmap_xyz", torch.from_numpy(envmap_xyz), persistent=False) # (env_h * env_w, 3)
        self.register_buffer("envmap_dir_wld", torch.from_numpy(envmap_dir_wld), persistent=False) # (env_h * env_w, 3)
        self.register_buffer("envmap_area", torch.from_numpy(envmap_area), persistent=False) # (env_h * env_w,)

        self.add_geometry_layer('texture_encoder', SpatialEncoder())
        if self.texture_encoder is not None:
            texture_feat_dim = self.texture_encoder.latent_size
        else:
            texture_feat_dim = 3

        hidden_feat_dim = 256
        texture_layers = 3

        pe_local_input_dim = 3

        self.pe_local_fn = NeRFPosEmbLinear(pe_local_input_dim, pe_local_input_dim * 6 * 2, no_linear=True, cat_input=True, no_pi=False) # posed space
        self.pe_canonical_fn = NeRFPosEmbLinear(3, 3 * 6 * 2, no_linear=True, cat_input=True, no_pi=False) # canonical space
        self.pe_ray_fn = NeRFPosEmbLinear(3, 3 * 4 * 2, no_linear=True, cat_input=True, angular=False, no_pi=False) # ray dir encoding

        pe_local_outdim = self.pe_local_fn.out_dim + self.pe_local_fn.in_dim
        pe_canonical_outdim = self.pe_canonical_fn.out_dim + self.pe_canonical_fn.in_dim
        pe_ray_outdim = self.pe_ray_fn.out_dim + self.pe_ray_fn.in_dim
        tex_input_dim = pe_ray_outdim + texture_feat_dim + 2 * hidden_feat_dim

        self.add_geometry_layer('merge_joint_field',
                FCBlock(hidden_feat_dim, 2, pe_local_outdim,
                        hidden_feat_dim, outermost_linear=False, with_ln=False))

        self.add_geometry_layer('skinning_deform',
                FCBlock(hidden_feat_dim, 0, hidden_feat_dim + texture_feat_dim, 3,
                        outermost_linear=True, with_ln=False))
        self.skinning_deform.net[-1].weight.data *= 0 # init deform block with zero output for stability
        self.skinning_deform.net[-1].bias.data *= 0

        # def slow_down_skinning_deform_grad(grad):
        #     return 0.1 * grad
        # self.skinning_deform.net[-1].weight.register_hook(slow_down_skinning_deform_grad)
        # self.skinning_deform.net[-1].bias.register_hook(slow_down_skinning_deform_grad)

        self.add_geometry_layer('feature_field',
                ImplicitField(pe_canonical_outdim, out_dim=hidden_feat_dim,
                        hidden_dim=hidden_feat_dim, num_layers=8, with_ln=False,
                        skips=[3], spec_init=True, use_softplus=False, experts=0))

        self.add_geometry_layer('predictor',
                SignedDistanceField(hidden_feat_dim, 128, num_layers=1,
                        with_ln=False, spec_init=True))

        self.add_radiance_layer('renderer',
                TextureField(tex_input_dim, hidden_feat_dim, texture_layers + 2,
                             with_ln=False, spec_init=True, experts=0))

        if self.relight:
            fixed_envmap_path = args.envmap_path
            self._load_environment_map(fixed_envmap_path, scale=self.envmap_factor, trainable=False,
                                       envmap_background_filepath=args.envmap_background)
            self.microfacet = Microfacet(f0=0.04)

            if args.static_albedo:
                # Load image with initial albedo prior
                # albedo_path = os.path.join(args.data_root, f"albedo_init_filter.png")
                albedo_path = None
                self._init_albedo_brdf_maps(args.static_albedo_res, args.static_albedo_res, albedo_path=albedo_path)

                # Build the delta net
                # uv_deform_feature_input_dim = pe_local_outdim + pe_canonical_outdim + texture_feat_dim
                uv_deform_feature_input_dim = pe_local_outdim + texture_feat_dim
                self.add_relight_layer('uv_deform',
                FCBlock(hidden_feat_dim, 3, uv_deform_feature_input_dim, 2,
                        outermost_linear=True, with_ln=False))
                self.uv_deform.net[-1].weight.data *= 0 # init deform block with small output for stability
                self.uv_deform.net[-1].bias.data *= 0

                # def slow_down_uv_deform_grad(grad):
                #     return torch.clamp(0.1 * grad, -0.5, 0.5)
                # self.uv_deform.net[-1].weight.register_hook(slow_down_uv_deform_grad)
                # self.uv_deform.net[-1].bias.register_hook(slow_down_uv_deform_grad)

            else:
                self.add_relight_layer('albedo_encoder', SpatialEncoder())
                albedo_feat_dim = self.albedo_encoder.latent_size
                albedo_feature_input_dim = albedo_feat_dim + 2 * hidden_feat_dim

                self.add_relight_layer('relight_renderer',
                        TextureField(albedo_feature_input_dim, hidden_feat_dim, texture_layers + 2,
                                    with_ln=False, spec_init=True, with_alpha=True, experts=0))

            if args.use_uv_nets:
                self.add_relight_layer('normalnet', NormalNet())
                if not self.use_uv_nets_norm_only:
                    self.add_relight_layer('visibilitynet', VisibilityNet())

        self.depth2normal = Depth2Normal()

        print ('NeuRA setup done')

    def add_geometry_layer(self, name, value):
        self.geometry_layers.append(name)
        setattr(self, name, value)

    def add_radiance_layer(self, name, value):
        self.radiance_layers.append(name)
        setattr(self, name, value)

    def add_relight_layer(self, name, value):
        self.relight_layers.append(name)
        setattr(self, name, value)

    def set_geometry_layers_trainable(self, trainable):
        for layer in self.geometry_layers:
            getattr(self, layer).requires_grad = trainable

    def set_radiance_layers_trainable(self, trainable):
        for layer in self.radiance_layers:
            getattr(self, layer).requires_grad = trainable


    def forward(self, rays_start, rays_dir, vertices,
                glob_rotation, glob_translation, joints_RT,
                mask=None, size=None, valid_verts=None, **kwargs):
        """Main forward function of the model in the validation mode.
        This function assumes that we handle a single pose (single frame) per call,
        which means that the batch size B=1.

        # Arguments
            rays_start: tensor with the start of the rays, with shape (B=1, C, 1, 3), C=num_crops
            rays_dir: tensor with the direction of each ray, with shape (B, C, num_rays, 3)
            vertices: global posed vertices of the human mesh, tensor with shape (B, V, 3)
            glob_rotation: tensor with shape (B, 3, 3)
            glob_translation: tensor with shape (B, 1, 3)
            joints_RT: tensor with shape (B, J=24, 4, 4)
            mask: tensor of matting mask with shape (B, H, W), or None
            size: image size as (W, H). If given, reshape the outputs to images patches.
            kwargs: dictionary with additional options:
                texture: tensor with shape (B, tex_h, tex_w, 3), normalized between [0,1]
                compute_normals: boolean, compute normal_map in the camera space
                    from estimated depth. Only supported if B=1.
                compute_uvmap_geometry: boolean, compute the geometry terms, e.i.,
                    normals and visibility w.r.t. the environment map, and projects
                    them to a UV space. `uv_normals`, `uv_vis`, and `uv_acc`
                    must be provided. Only supported if B=1.
                compute_visibility: 
                uv_normals: tensor that buffers the geometry normals, shape (B, tex_h * tex_w, 3)
                uv_tex: tensor that buffers the texture map, shape (B, 4 * tex_h * tex_w, 3)
                uv_vis: tensor that buffers the visibility flags, shape (B, tex_h * tex_w, num_env_pix)
                rgb_tgt: target RGB image with size (B, C, num_rays, 3), only used to compute the texture maps

        # Returns
        """
        batch = rays_start.shape[0]
        if len(rays_start.shape) == 4:
            assert batch == 1, (f'Invalid batch size (batch={batch})!')
            rays_start = rays_start.squeeze(0)
            rays_dir = rays_dir.squeeze(0)

        if valid_verts is not None:
            valid_verts = valid_verts.squeeze(0)

        vertices = vertices.squeeze(0)
        glob_rotation = glob_rotation.squeeze(0)
        glob_translation = glob_translation.squeeze(0)
        joints_RT = joints_RT.squeeze(0)
        if mask is not None:
            mask = mask.squeeze(0)
        for key in kwargs.keys():
            if (kwargs[key] is not None) and (isinstance(kwargs[key], torch.Tensor)):
                kwargs[key] = kwargs[key].squeeze(0)

        outputs = {}

        def if_size_reshape(x, ch=None):
            if (x is not None) and (size is not None):
                W, H = size
                if ch is not None:
                    return x.view(-1, H, W, ch)
                return x.view(-1, H, W)
            return x

        ######################################################################
        # Compute CNN-based features from RGB texture input
        ######################################################################
        texture_rgb = kwargs['texture']
        if len(texture_rgb.shape) == 3:
            texture_rgb = texture_rgb.unsqueeze(0)

        # Arrange the texture map as from (tex_h, tex_w, 3) to (1, 3, tex_h, tex_w) and flip.
        texture_rgb = texture_rgb.permute(0, 3, 1, 2) # (1, 3, tex_h, tex_w)
        texture_rgb = torch.flip(texture_rgb, dims=[2])
        texture_map = texture_rgb
        if self.texture_encoder is not None:
            texture_map = self.texture_encoder(texture_map) # texture_rgb from [-1..1] to [0..1]

        # Check if we are using uv_normals and visibility, and we are not computing them
        if self.use_uv_nets:
            uv_normals_input = kwargs['uv_normals_input'].unsqueeze(0).permute(0, 3, 1, 2) # (1, 3, tex_h, tex_w)
            uv_normals_input_mask = kwargs['uv_normals_input_mask'].unsqueeze(0).unsqueeze(1) # (1, 1, tex_h, tex_w)
            uv_normals_pred, uv_normals_mask = self.normalnet(uv_normals_input, uv_normals_input_mask)
            uv_normals = normalize_fn(torch.flip(uv_normals_pred, dims=[2]), axis=1)[0]
            # uv_normals = kwargs['uv_normals_dense'].unsqueeze(0).permute(0, 3, 1, 2) # (1, 3, tex_h, tex_w)
            # uv_normals_pred = uv_normals
            # uv_normals = normalize_fn(torch.flip(uv_normals, dims=[2]), axis=1)[0]
        elif ('uv_normals_dense' in kwargs) and ('compute_uvmap_geometry' not in kwargs):
            uv_normals = kwargs['uv_normals_dense'].unsqueeze(0).permute(0, 3, 1, 2) # (1, 3, tex_h, tex_w)
            uv_normals = normalize_fn(torch.flip(uv_normals, dims=[2]), axis=1)[0]
        else:
            uv_normals = None

        if self.use_uv_nets and not self.use_uv_nets_norm_only:
            uv_vis_input = kwargs['uv_vis_input'].unsqueeze(0).permute(0, 3, 1, 2) # (1, 512, tex_h, tex_w)
            uv_vis_input_mask = kwargs['uv_vis_input_mask'].unsqueeze(0).unsqueeze(1) # (1, 1, tex_h, tex_w)
            uv_vis_pred, uv_vis_mask = self.visibilitynet(uv_vis_input, uv_vis_input_mask, uv_normals_input, uv_normals_input_mask)
            uv_vis = torch.flip(uv_vis_pred, dims=[2])
            # uv_vis = kwargs['uv_vis_dense'].unsqueeze(0).permute(0, 3, 1, 2) # (1, num_lights, tex_h, tex_w)
            # uv_vis_pred = uv_vis
            # uv_vis = torch.clamp(torch.flip(uv_vis, dims=[2]), 0, 1)
        elif ('uv_vis_dense' in kwargs) and ('compute_uvmap_geometry' not in kwargs):
            uv_vis = kwargs['uv_vis_dense'].unsqueeze(0).permute(0, 3, 1, 2) # (1, num_lights, tex_h, tex_w)
            uv_vis = torch.clamp(torch.flip(uv_vis, dims=[2]), 0, 1)
            uv_vis_pred = uv_vis
        else:
            uv_vis = None

        # if ('uv_albedo' in kwargs):
        #     uv_albedo = kwargs['uv_albedo'].unsqueeze(0).permute(0, 3, 1, 2) # (1, 3, tex_h, tex_w)
        #     uv_albedo = torch.flip(uv_albedo, dims=[2])
        # else:
        #     uv_albedo = None

        intersec_out = self.ray_intersect_point_cloud(rays_start, rays_dir,
                vertices, filter_valid_rays=not self.training, mask=mask,
                valid_verts=valid_verts)

        rays_hits = intersec_out['mesh_hits']

        valid_rays_start = torch.tile(rays_start, (1, rays_dir.shape[1], 1))
        valid_rays_start = valid_rays_start[rays_hits] # (N, 3), N = all valid rays
        valid_rays_dir = rays_dir[rays_hits] # (N, 3)

        mesh_depth_min = intersec_out['mesh_depth_min'][rays_hits] # (N, 1)
        mesh_depth_max = intersec_out['mesh_depth_max'][rays_hits] # (N, 1)

        # New implementation
        triangles = F.embedding(self.faces, vertices) # (F, 3, 3)
        face_weights = F.embedding(self.faces, self.skinning_weights)  # (F, 3, 24)
        raymarcher_out = self.mesh_based_ray_marcher(
                rays_start=valid_rays_start,
                rays_dir=valid_rays_dir,
                depth_min=mesh_depth_min,
                depth_max=mesh_depth_max,
                max_num_steps=self.num_steps,
                triangles=triangles,
                face_weights=face_weights,
                texture_map=texture_map,
                glob_rotation=glob_rotation,
                glob_translation=glob_translation,
                joints_RT=joints_RT,
                uv_normal_map=uv_normals,
                uv_visibility_map=uv_vis,
                relight=self.relight)

        if 'compute_normals' in kwargs and kwargs['compute_normals']:
            assert batch == 1, (f'For normals computation, `B!=1` is not supported (B={batch})!')
            # normal_map, surface_xyz = compute_normal_map(valid_rays_start[:1], valid_rays_dir, raymarcher_out['depth'], rays_hits, size)

            # surface_depth = masked_scatter(rays_hits, raymarcher_out['depth'])[0] # (N, 3)
            # Remove 0.5 cm from the depth to make sure the surface is not inside the neural field.
            # This has no effect in the normals, but helps in the visibility later
            surface_depth = masked_scatter_value(rays_hits, raymarcher_out['depth'] - 0.005, MAX_DEPTH)[0] # (N, 3)

            normal_map, surface_xyz = self.depth2normal(rays_start[0], rays_dir[0], surface_depth.unsqueeze(-1), size)
            # normal_map, surface_xyz = compute_normal_map2(rays_start[0], rays_dir[0], surface_depth.unsqueeze(-1), size)

            outputs['normals'] = normal_map.unsqueeze(0)

            if 'compute_uvmap_geometry' in kwargs and kwargs['compute_uvmap_geometry']:
                assert batch == 1, (f'For normals computation, `B!=1` is not supported (B={batch})!')
                assert 'uv_normals' in kwargs, (f'For normals computation in UV, `uv_normals` holder is required')
                assert 'uv_vis' in kwargs, (f'For visibility computation in UV, `uv_vis` holder is required')

                valid_hits = raymarcher_out['prob_hit'].gt(0.99) # (N,)

                # Here we discard rays where the angle between normal and camera ray is higher
                # than 60 degrees. For those (discarded) points, it is very likely that the
                # normal is wrong, and/or the point in on the tangent of the surface
                valid_normal = torch.einsum("nd,nd->n", normal_map.view(-1, 3), rays_dir[0])
                valid_hits *= valid_normal[rays_hits[0]].lt(-0.707) # cos(60)

                valid_sampled_rays = masked_scatter(rays_hits, valid_hits) # (1, num_rays)

                pose_data = {
                    'vertices': vertices,
                    'texture_map': texture_map,
                    'glob_rotation': glob_rotation,
                    'glob_translation': glob_translation,
                    'joints_RT': joints_RT,
                }

                self.update_uvmap_geometry(
                    grids=raymarcher_out['grids'][valid_hits],
                    normals=normal_map.view(-1, 3)[valid_sampled_rays[0]],
                    rgb=None,
                    # rgb=kwargs['rgb_tgt'][rays_hits[0]][valid_hits], #raymarcher_out['rgb'][valid_hits],
                    surface_xyz=surface_xyz.view(-1, 3)[valid_sampled_rays[0]],
                    tex_size=texture_rgb.shape[2:][::-1],
                    env_map_size=self.envmap_size,
                    inputs=kwargs,
                    pose_data=pose_data,
                    compute_visibility=kwargs['compute_visibility'])

        # convert rgb colors to full buffer
        prob_hit = masked_scatter(rays_hits, raymarcher_out['prob_hit'])
        rgb = masked_scatter(rays_hits, raymarcher_out['rgb'])
        relight = masked_scatter(rays_hits, raymarcher_out['relight']) if self.relight else None
        hdr = masked_scatter(rays_hits, raymarcher_out['hdr']) if self.relight else None
        albedo = masked_scatter(rays_hits, raymarcher_out['albedo']) if self.relight else None
        rough = masked_scatter(rays_hits, raymarcher_out['rough']) if self.relight else None
        uv_residual = masked_scatter(rays_hits, raymarcher_out['uv_residual']) if self.relight else None
        sigma = masked_scatter(rays_hits, raymarcher_out['sigma'])
        probs = masked_scatter(rays_hits, raymarcher_out['probs'])
        grids = masked_scatter(rays_hits, raymarcher_out['grids']) if raymarcher_out['grids'] is not None else None
        depth = masked_scatter_value(rays_hits, raymarcher_out['depth'], MAX_DEPTH) if raymarcher_out['depth'] is not None else None
        residual = raymarcher_out['residual'].view(-1, 3) if raymarcher_out['residual'] is not None else None

        outputs.update({
            'prob_hit': if_size_reshape(prob_hit),
            'rgb': if_size_reshape(rgb, ch=3),
            'sigma': if_size_reshape(sigma, ch=sigma.size(-1)),
            'probs': if_size_reshape(probs, ch=probs.size(-1)),
            'grids': if_size_reshape(grids, ch=2),
            'depth': if_size_reshape(depth),
            'residual': residual,
            'relight': if_size_reshape(relight, ch=3) if self.relight else None,
            'hdr': if_size_reshape(hdr, ch=3) if self.relight else None,
            'albedo': if_size_reshape(albedo, ch=3) if self.relight else None,
            'rough': if_size_reshape(rough, ch=1) if self.relight else None,
            'uv_residual': if_size_reshape(uv_residual, ch=2) if self.relight else None,
            'sampled_uv_residual': raymarcher_out['sampled_uv_residual'] if self.relight else None,
            })

        if self.use_uv_nets:
            outputs.update({'uv_normals_pred': uv_normals_pred})
            if not self.use_uv_nets_norm_only:
                outputs.update({'uv_vis_pred': uv_vis_pred})

            if not self.training: # useful for debug
                outputs.update({
                    'uv_normals_input': uv_normals_input,
                    'uv_vis_input': uv_vis_input,
                    'uv_normals_pred_mask': uv_normals_mask,
                    'uv_vis_pred_mask': uv_vis_mask,
                    })

        if not self.training:
            outputs.update({
                'mesh_hits': if_size_reshape(rays_hits),
            })

        for key in ['sampled_normals', 'sampled_visibility', 'sampled_tex_rgb', 'local_coords']:
            if key in raymarcher_out and raymarcher_out[key] is not None:
                _d = masked_scatter(rays_hits, raymarcher_out[key])
                outputs[key] = if_size_reshape(_d, ch=_d.size(2))

        #####################################################
        ## Render background HDR image if needed
        #####################################################
        if ('full_lat_lng' in kwargs) and (hasattr(self, 'envmap_background')):
            outputs['bground_pix'] = self.render_background_envmap(kwargs['full_lat_lng'])

        return outputs


    def render_background_envmap(self, full_lat_lng):
        """Render the background pixels from `envmap_background` -- an HDRi
        in lat-lng coordinates.

        # Arguments:
            full_lat_lng: tensor with shape (N, 2)

        # Returns
            Tensor with RGB values (after tonemaping) with shape (N, 3)
        """
        assert hasattr(self, 'envmap_background'), (f'`envmap_background` not set!')

        # Convert from OpenEXR "latlong" format to U-V grid mapping
        v = -full_lat_lng[:, 0] / (np.pi / 2) # convert from [pi/2, -pi/2] to [-1, 1]
        u = -full_lat_lng[:, 1] / np.pi # convert from [pi, -pi] to [-1, 1]
        grids = torch.stack([u, v], dim=-1)[None, :, None, :] # (1, N, 1, 2)
        hdri = self.envmap_background.permute(2, 0, 1)[None, ...] # (B=1, 3, H, W)
        sampled_hdr = F.grid_sample(hdri, grids, mode='nearest', align_corners=False)  # (1, 3, N, 1)
        sampled_hdr = sampled_hdr[0, :, :, 0].permute(1, 0)
        # rgb = sampled_hdr / (1 + sampled_hdr)
        rgb = func_linear2srgb(sampled_hdr)

        # DEBUG:: show the camera rays in obj
        # rays_dir_xyz = 1000 * full_rays_dir
        # rays_dir_color = torch.ones_like(rays_dir_xyz)
        # rays_dir_color[:, 1:3] *= 0

        # filename = f'./output/debug_full_rays_dir.obj'
        # save_color_pc_obj(filename, rays_dir_xyz.detach().cpu().numpy(), rays_dir_color.detach().cpu().numpy())

        # DEBUG:: convert the lat-lng format back to xyz to compare with the original camera rays
        # full_r_lat_lng = torch.cat([1000 * torch.ones_like(full_lat_lng[:, 0:1]), full_lat_lng], dim=-1)
        # lat_lng_color = torch.ones_like(full_r_lat_lng)
        # lat_lng_color[:, 0:2] *= 0

        # full_xyz_reverse =  sph2cart(full_r_lat_lng.detach().cpu().numpy())
        # filename = f'./output/debug_full_lat_lng.obj'
        # save_color_pc_obj(filename, full_xyz_reverse, lat_lng_color.detach().cpu().numpy())

        # TODO: use grids here to set the pixels in the env map for debug -> new zeros, then set sampled pixels to red

        return rgb


    def ray_intersect_point_cloud(self, rays_start, rays_dir, vertices,
            filter_valid_rays=True, mask=None, valid_verts=None):
        """This function intersects a set of rays with the explicit human mesh.
        The batch dimension B has to be 1 when `filter_valid_rays=True`.

        # Arguments
            rays_start: tensor with shape (C, 1, 3)
            rays_dir: tensor with shape (C, num_rays, 3)
            vertices: tensor with shape (V, 3)
            texture_rgb: tensor with shape (B=1, 3, h_tex, w_tex)
            mask: tensor with shape (C, H, W)
            filter_valid_rays: boolean, whether to filter only valid rays before casting them.
                It is not very useful if the rays are already defined in the surface.
                If the `mask` is given, use the mask to define which rays are valid. Otherwise,
                intersect all the rays with a sphere centered in the person.
                Usually used at inference only.

        # Returns
            Check out at the end of the function.
        """
        batch = rays_start.shape[0]
        vertices = vertices.unsqueeze(0)
        if filter_valid_rays:
            assert (batch == 1), (
                f'Batch size `!= 1` is not supported (B={batch}) when filter_valid_rays={filter_valid_rays}!')

        if filter_valid_rays:
            # stores the depth where each ray hits something
            mesh_depth_min = MAX_DEPTH * torch.ones_like(rays_dir[..., 0], device=rays_dir.device) # (B, num_rays)
            mesh_depth_max = mesh_depth_min.clone()

            if mask is not None:
                img_mask = (mask > 0.5).view((-1,))
                valid_rays = img_mask.unsqueeze(0) # (B, num_rays)
            else:
                body_center = vertices.mean(1, keepdim=True) # average of all body vertices (B, 1, 3)
                # Defined the expected ratio of a sphere where the human is enclosed
                body_max_radius, _ = (vertices - body_center).norm(2, -1).max(dim=1, keepdim=True)
                body_max_radius = 1.25 * body_max_radius # farthest joint from the center (B,)
                # we intersect each set of rays with the bounding sphere
                h_cet = body_center - rays_start # (B, 1, 3)
                h_mid = (h_cet * rays_dir).sum(-1) # (B, num_rays)
                h_dis = (h_cet ** 2).sum(-1) - h_mid ** 2 # == closest_point_to_the_center ** 2 (B, num_rays)
                valid_rays = h_dis < (body_max_radius ** 2) # (B, num_rays)
        
            valid_rays_dir = rays_dir[:, valid_rays[0]] # here B!=1 is not supported
        else:
            valid_rays_dir = rays_dir

        # we intersect each set of remaining rays with the mesh
        rays_start_tiled = torch.tile(rays_start, (1, valid_rays_dir.shape[1], 1))

        if valid_verts is not None:
            assert (valid_verts.size(0) == rays_start_tiled.size(0)), (
                f'`valid_verts` must match `rays_start_tiled`, '
                f'but {valid_verts.size(0)} != {rays_start_tiled.size(0)}')
            hit_cloud = []
            hit_min_depth = []
            hit_max_depth = []

            for i in range(valid_verts.size(0)):
                aux = vertices[:, valid_verts[i]]
                _hit_cloud, _hit_min_depth, _hit_max_depth = cloud_ray_intersect(
                        self.min_dis_eps,
                        aux, # (?, V', 3)
                        rays_start_tiled[i:i+1], # (1, num_rays, 3)
                        valid_rays_dir[i:i+1]) # (1, num_rays, 3)
                hit_cloud.append(_hit_cloud)
                hit_min_depth.append(_hit_min_depth)
                hit_max_depth.append(_hit_max_depth)

            hit_cloud = torch.concat(hit_cloud, dim=0)
            hit_min_depth = torch.concat(hit_min_depth, dim=0)
            hit_max_depth = torch.concat(hit_max_depth, dim=0)
            hit_cloud = hit_cloud.squeeze(2).bool() # (B, num_rays)

        else:
            # Fake a batch=1 to avoid OOM in triangle_ray_intersect
            rays_start_tiled = rays_start_tiled.view(-1, 3).unsqueeze(0)
            valid_rays_dir = valid_rays_dir.view(-1, 3).unsqueeze(0)

            # Intersect the rays with the mesh point cloud given a margin
            hit_cloud, hit_min_depth, hit_max_depth = cloud_ray_intersect(
                    self.min_dis_eps,
                    vertices, # (1, V, 3)
                    rays_start_tiled, # (1, B*num_rays, 3)
                    valid_rays_dir) # (1, B*num_rays, 3)

            hit_cloud = hit_cloud.view(batch, -1, 1)
            hit_min_depth = hit_min_depth.view(batch, -1, 1)
            hit_max_depth = hit_max_depth.view(batch, -1, 1)
            hit_cloud = hit_cloud.squeeze(2).bool() # (B, num_rays)

        if filter_valid_rays: # TODO FIXME continue
            a = hit_cloud.unsqueeze(2).float()

            _depth_min = a * hit_min_depth + (1 - a) * MAX_DEPTH
            _depth_min[hit_cloud] = hit_min_depth[hit_cloud]
            mesh_depth_min[valid_rays] = _depth_min.squeeze(-1)
            

            _depth_max = a * hit_max_depth
            _depth_max[hit_cloud] = hit_max_depth[hit_cloud]
            mesh_depth_max[valid_rays] = _depth_max.squeeze(-1)

            mesh_hits = valid_rays.clone()
            mesh_hits[valid_rays] = hit_cloud # shape (B, num_rays)
        else:
            mesh_depth_min = hit_min_depth.squeeze(-1)
            mesh_depth_max = hit_max_depth.squeeze(-1)

            mesh_hits = hit_cloud

        mesh_color = None

        intersection_outputs = {
            "mesh_depth_min": mesh_depth_min.unsqueeze(-1), # (B, num_rays, 1)
            "mesh_depth_max": mesh_depth_max.unsqueeze(-1), # (B, num_rays, 1)
            "mesh_hits": mesh_hits, # (B, num_rays)
            "mesh_color": mesh_color, # (B, num_rays, 3)
        }

        return intersection_outputs


    def intersect_points_with_mesh(self,
                                   sampled_xyz,
                                   triangles,
                                   texture_map,
                                   face_weights,
                                   glob_rotation,
                                   glob_translation,
                                   joints_RT,
                                   cuda_version=True,
                                   topk=1,
                                   uv_normal_map=None,
                                   uv_visibility_map=None,
                                   **kwargs):
        """
        # Arguments
            sampled_xyz: tensor with shape (N=num_rays, 3)
            triangles: shape (F, 3, 3)
            texture_map: shape (B=1, f_dim, feat_h, feat_w)
            face_weights: shape (F, 3, J=24)
            glob_rotation: shape (3, 3)
            glob_translation: shape (1, 3)
            joints_RT: shape (J=24, 4, 4)
            uv_normal_map: normal map (batch=1, 3, H, W)
            uv_visibility_map: visibility map (batch=1, num_lights, H, W)

        # Returns
            A dictionary. Check the implementation!
        """
        if cuda_version is False:
            print (f"Warning! (cuda_version={cuda_version}): This is an experimental (and not working) implementation!")

        outs = {}
        l_idx = torch.tensor([0,]).type_as(self.faces) # (1,)
        face_uv = self.face_uv
        f_dim = texture_map.size(1)

        # TODO: Compute the distance from sampled_xyz to the triangles
        if cuda_version:
            # Intersect points with the mesh. Returns all tensors with shape (N,)
            min_dis, min_face_idx, w0, w1, w2 = point_face_dist_forward(
                sampled_xyz, l_idx, triangles, l_idx, sampled_xyz.size(0))
            outs['min_dis'] = min_dis.sqrt()

            # Sample from the texture map
            bary_coords = torch.stack([w0, w1, w2], 1)   # (N, 3)
            sampled_uvs = (face_uv[min_face_idx] # (N, 3, 2)
                    * bary_coords.unsqueeze(-1) # (N, 3, 1)
                    ).sum(1) # (N, 2)
            grids = sampled_uvs[None, :, None, :] * 2 - 1   # UV coords normalized as [-1, 1] with shape (1, N, 1, 2)
            sampled_tex = F.grid_sample(texture_map, grids, mode='bilinear', align_corners=False)  # (1, f_dim, N, 1)
            outs['sampled_tex'] = sampled_tex.permute(0, 2, 3, 1).reshape(-1, f_dim)  # (N, f_dim)

            # if uv_normal_map is not None:
            #     sampled_normals = F.grid_sample(uv_normal_map, grids, mode='bilinear', align_corners=False)  # (1, 3, N, 1)
            #     outs['sampled_normals'] = sampled_normals.permute(0, 2, 3, 1).reshape(-1, 3)  # (N, 3)

            # if uv_visibility_map is not None:
            #     num_lights = uv_visibility_map.size(1)
            #     sampled_visibility = F.grid_sample(uv_visibility_map, grids, mode='bilinear', align_corners=False)  # (1, num_lights, N, 1)
            #     outs['sampled_visibility'] = sampled_visibility.permute(0, 2, 3, 1).reshape(-1, num_lights)  # (N, num_lights)

            # Compute the local coordinates
            A, B, C = triangles[:, 0], triangles[:, 1], triangles[:, 2]
            triangle_normals = torch.cross((B - A), (C - B), 1) # (N, 3)
            triangle_normals_mag = triangle_normals.norm(dim=1)
            faces_xyzs = (triangles[min_face_idx] * bary_coords.unsqueeze(-1)).sum(1) # (N, 3)
            insideout = ((sampled_xyz - faces_xyzs) * triangle_normals[min_face_idx]).sum(-1).sign() #  [-1, 1] (N,)
            insideout = insideout * triangle_normals_mag.ge(1e-6).float()[min_face_idx]

        else:
            centroids = torch.mean(triangles, dim=1) # (F, 3)
            dist = torch.sum((sampled_xyz.unsqueeze(1) - centroids.unsqueeze(0)) ** 2, dim=-1)
            min_dis, topk_faces = torch.topk(dist, k=topk, dim=1, largest=False)

            outs['min_dis'] = 0
            outs['sampled_tex'] = 0
            sampled_uvs = 0
            insideout = 0
            for k in range(topk):
                min_face_idx = topk_faces[:, k]
                w0, w1, w2, _insideout = compute_barycentric_coordinates(sampled_xyz, triangles[min_face_idx])
                insideout = insideout + _insideout
                outs['min_dis'] = outs['min_dis'] + torch.mean(min_dis.sqrt(), dim=1)

                bary_coords = torch.stack([w0, w1, w2], 1)   # (N, 3)
                _sampled_uvs = (face_uv[min_face_idx] # (N, 3, 2)
                        * bary_coords.unsqueeze(-1) # (N, 3, 1)
                        ).sum(1) # (N, 2)
                sampled_uvs = sampled_uvs + _sampled_uvs
                grids = _sampled_uvs[None, :, None, :] * 2 - 1   # UV coords normalized as [-1, 1] with shape (1, N, 1, 2)
                sampled_tex = F.grid_sample(texture_map, grids, mode='bilinear', align_corners=False)  # (1, f_dim, N, 1)
                outs['sampled_tex'] = outs['sampled_tex'] + sampled_tex.permute(0, 2, 3, 1).reshape(-1, f_dim)  # (N, f_dim)

            outs['min_dis'] /= topk
            outs['sampled_tex'] /= topk
            sampled_uvs /= topk
            insideout /= topk

        outs['local_coords'] = torch.cat([
                sampled_uvs * 2 - 1, # (N, 2)
                (outs['min_dis'] * insideout)[:, None] # / (2 * self.min_dis_eps) # (N, 1)
            ], 1) # (N, 3)

        # TODO: replace the face-based skinning by a vertices weights skinning
        # This causes artifacts in the 
        # Map the 3D points from posed space to canonical space using the blending weights
        weights = (face_weights[min_face_idx] * bary_coords.unsqueeze(-1)).sum(1) # (N, 24)
        # weights = (face_weights[min_face_idx]).sum(1) # (N, 24)
        A_inv = torch.inverse(joints_RT).reshape(-1, 16)

        weighted_RT_inv = (weights @ A_inv).reshape(-1, 4, 4) # (N, 4, 4)
        sampled_xyz_local = torch.matmul(sampled_xyz - glob_translation, glob_rotation.T)
        sampled_xyz_local_homo = torch.cat([sampled_xyz_local,
                sampled_xyz_local.new_ones(sampled_xyz_local.size(0), 1)], dim=-1)  # homogeneous coords (N, 4)
        sampled_xyz_canonical = torch.einsum("ncd,nd->nc", weighted_RT_inv, sampled_xyz_local_homo)

        if self.tpose_RT is not None: # A custom T-Pose is given
            tpose_weighted_RT = (weights @ self.tpose_RT).reshape(-1, 4, 4) # (N, 4, 4)
            sampled_xyz_canonical = torch.einsum("ncd,nd->nc", tpose_weighted_RT, sampled_xyz_canonical)

        canonical_coords = sampled_xyz_canonical[:, :3] # Final points in the canonical space, i.e., unposed (N, 3)
        if self.canonical_mesh_bounds_min is not None:
            # Normalize canonical coordinates based on the canonical mesh bounds (if given)
            avg_center = (self.canonical_mesh_bounds_max + self.canonical_mesh_bounds_min) / 2
            max_size = (self.canonical_mesh_bounds_max - self.canonical_mesh_bounds_min).max()
            canonical_coords = (2 * (canonical_coords - avg_center[None, :]) / max_size) / 1.1

        outs['canonical_coords'] = canonical_coords
        outs['grids'] = grids.squeeze(2).squeeze(0) # shape (N, 2) interval [-1, 1]

        return outs


    def _mesh_based_ray_marcher_chunk(self,
                               rays_start,
                               rays_dir,
                               depth_min,
                               depth_max,
                               max_num_steps,
                               triangles,
                               face_weights,
                               texture_map,
                               glob_rotation,
                               glob_translation,
                               joints_RT,
                               density_only=False,
                               relight=False,
                               **kwargs):
        """This function performs a ray marching algorithm in a neural field
        guided by an explicit mesh. The ray is defined in the following manner:

        rays_start (3D) x----------------------------> rays_dir (3D)
                        |                            |
                    depth_min                   depth_max

        Points will be sampled along the ray, the distance between each points
        is `(depth_max - depth_min) / max_num_steps`. When a sampled point
        is too far from the vertices of the mesh (> `self.min_dis_eps`), it
        will be skipped, assuming zero density. In total, up to
        `S = max_num_steps + 1` points will be sampled along each ray.

        # Arguments
            rays_start, rays_dir: tensor with shape (num_rays, 3)
            depth_min, depth_max: tensors with shape (num_rays, 1)
            triangles: shape (F, 3, 3)
            face_weights: shape (F, 3, 24)
            texture_map: shape (B=1, f_dim, feat_h, feat_w)
            glob_rotation: shape (3, 3)
            glob_translation: shape (1, 3)
            joints_RT: shape (J=24, 4, 4)
            density_only: boolean, skips color if True
            relight: boolean, if performing inverse rendering
            kwargs: dictionary
                'uv_normal_map': input UV map with normals (batch, 3, H, W),
                'uv_visibility_map': input UV map with visibility (batch, num_lights, H, W),

        # Returns

        """
        S = max_num_steps + 1
        N = rays_start.shape[0]
        sigma_agg = rays_start.new_zeros(size=(N, S))
        rgb_agg = rays_start.new_zeros(size=(N, S, 3))
        grids_agg = rays_start.new_zeros(size=(N, S, 2))
        # localcoords_agg = rays_start.new_zeros(size=(N, S, 3))
        rays_mask = torch.ones((N,), dtype=torch.bool, device=rays_start.device)
        step_check = 4

        if self.training:
            canonical_residual_agg = rays_start.new_zeros(size=(N, S, 3))
        else:
            canonical_residual_agg = None

        # Compute the depth of each sampled point in each ray
        depth_step = torch.tile(((depth_max - depth_min) / max_num_steps)[:, None], (1, S, 1))
        if self.training:
            depth_step = torch.clamp(depth_step + torch.zeros_like(depth_step).normal_(0, 1e-3), 1e-5, None)
        # accumulate steps and add depth offset (start of the ray)
        sampled_depth = torch.cumsum(depth_step, dim=1) + depth_min.unsqueeze(1)

        def _aggregate_samples(step):
            nonlocal rays_mask

            sigmap = torch.relu(sigma_agg[:, :step + 1])
            free_energy = sigmap * depth_step[:, :step + 1].squeeze(2)
            shifted_free_energy = torch.cat([
                    free_energy.new_zeros(N, 1),
                    free_energy[:, :-1]
                ], dim=-1)  # shift one step, (num_rays, num_steps+1)
            a = 1 - torch.exp(-free_energy.float()) # probability of it is not empty here (N, S') -> S' steps so far
            b = torch.exp(-torch.cumsum(shifted_free_energy.float(), dim=-1)) # probability of everything is empty up to now (N, S')
            probs = (a * b).type_as(free_energy) # (N, S')
            prob_hit = probs.sum(-1) # Chance that the ray did intersect something dense (N,)
            rays_mask *= prob_hit.lt(0.999)

            rgb = None
            if not density_only:
                rgb = (rgb_agg[:, :step + 1] * probs.unsqueeze(-1)).sum(1)

            return prob_hit, rgb, probs


        for step in range(S):
            sampled_xyz = ray(rays_start, rays_dir, sampled_depth[:, step])
            if self.training:
                sampled_xyz += torch.zeros_like(sampled_xyz).normal_(0, 2e-4)

            intersec_data = self.intersect_points_with_mesh(sampled_xyz,
                    triangles, texture_map, face_weights, glob_rotation,
                    glob_translation, joints_RT)

            # Discard points that are too far from the mesh
            sampled_pts_mask = torch.abs(intersec_data['min_dis']) < self.min_dis_eps
            sampled_pts_mask *= rays_mask # check the general ray mask
            sampled_dir = rays_dir[sampled_pts_mask]
            if sampled_dir.size(0) == 0:
                if ((step + 1) % step_check == 0) or (step == S - 1) or (rays_mask.max() == 0):
                    prob_hit, rgb, probs = _aggregate_samples(step)
                if rays_mask.max() == 0:
                    break # we are done with all the rays
                continue # we still have valid non-hit rays running out

            _sigma, _rgb, _residual = self._ray_marcher_inner_radiance_field(
                    sampled_dir,
                    intersec_data['local_coords'][sampled_pts_mask],
                    intersec_data['sampled_tex'][sampled_pts_mask],
                    intersec_data['canonical_coords'][sampled_pts_mask],
                    density_only=density_only)

            sigma_buf = rays_start.new_zeros(size=(N,))
            sigma_buf[sampled_pts_mask] = _sigma
            sigma_agg[:, step] = sigma_buf

            if not density_only:
                rgb_buf = rays_start.new_zeros(size=(N, 3))
                rgb_buf[sampled_pts_mask] = _rgb
                rgb_agg[:, step] = rgb_buf

                grids_agg[:, step] = intersec_data['grids']
                # localcoords_agg[:, step] = intersec_data['local_coords']

            if self.training:
                canonical_residual_buf = rays_start.new_zeros(size=(N, 3))
                canonical_residual_buf[sampled_pts_mask] = _residual
                canonical_residual_agg[:, step] = canonical_residual_buf

            # In every 4 iterations (steps) or in the last one
            if ((step + 1) % step_check == 0) or (step == S - 1):
                # Check whether we have a high probability of hitting something so far.
                # If so, then disable the ray by masking it out.
                prob_hit, rgb, probs = _aggregate_samples(step)

        depth = None
        grids = None

        # print (f'DEBUG:: prob_hit.shape, prob_hit.max()', prob_hit.shape, prob_hit.max())

        # For the non-missed rays, re-normalize probs. Only at inference time or for relightning.
        if (not self.training) or relight:
            # resample along the ray with fewer samples
            if relight:
                rel_hit = prob_hit.gt(0.01)
                # print (f'  DEBUG:: rel_hit.shape, rel_hit.max()', rel_hit.shape, rel_hit.max())
                # if rel_hit.max() < 1: # hack to avoid empty hel_hit
                #     rel_hit = prob_hit.gt(prob_hit.mean())
            else:
                rel_hit = prob_hit.gt(0.5)
            not_missed = rel_hit.unsqueeze(1).float() # (num_rays, 1)

            if probs.size(1) < S:
                probs = torch.cat([probs, probs.new_zeros(size=(probs.size(0), S - probs.size(1)))], dim=1)
            probs_nm = (probs / probs.sum(-1, keepdim=True).clamp(1e-4)) # (num_rays, S)
            # depth = (sampled_depth.squeeze(-1) * (not_missed * probs_nm)).sum(-1) # Expected depth values (num_rays,)
            depth = ( # Expected depth values (num_rays,)
                (sampled_depth.squeeze(-1) * (not_missed * probs_nm)).sum(-1)
                + (1 - not_missed.squeeze(1)) * MAX_DEPTH
            )

            if relight:
                # Compute the boundary in depth for recasting rays for relighting
                cdf = torch.cumsum(probs_nm, dim=1) # (N, S)
                hit1 = cdf.gt(0.001).float()
                hit2 = cdf.gt(0.999).float()
                hit1_idx = torch.argmax(hit1[:, 1:] - hit1[:, :-1], dim=1, keepdim=True) # (N, 1)
                hit2_idx = torch.clamp(torch.argmax(hit2[:, 1:] - hit2[:, :-1], dim=1, keepdim=True) + 1, 0, max_num_steps)
                depth_min = torch.gather(sampled_depth.squeeze(2), index=hit1_idx, dim=1) # (N, S) -> (N,)
                depth_max = torch.gather(sampled_depth.squeeze(2), index=hit2_idx, dim=1) # (N, S) -> (N,)

            ## Debug: how local coordinates
            # local_coords = (localcoords_agg * (not_missed * probs_nm).unsqueeze(-1)).sum(1) # Expected depth values (num_rays,)

            max_prob_idx = torch.argmax(probs, dim=1, keepdim=True).unsqueeze(2) # (num_rays, 1, 1)
            grids = torch.gather(grids_agg, index=max_prob_idx.tile(1, 1, 2), dim=1).squeeze(1) # (num_rays, steps+1, 2) -> # (num_rays, 2)
            ## Soft version, does not work well in practice
            # grids = (grids_agg * (not_missed * probs_nm).unsqueeze(2)).sum(1) # (num_rays, steps+1, 2) -> # (num_rays, 2)

        raymarcher_out = {
            'prob_hit': prob_hit, # (N,)
            'rgb': rgb, # (N, 3)
            'depth': depth, # (N,)
            'sigma': sigma_agg, # (N, S)
            'probs': probs, # (N, S)
            'residual': canonical_residual_agg, # (N, S, 3)
            'grids': grids, # (N, 2)
            # 'local_coords': local_coords, # (N, 3)
        }

        # If relighting, recast the rays with inverse rendering
        if relight:
            relight_S=16

            # print (f'DEBUG:: rel_hit.shape, rel_hit.max()', rel_hit.shape, rel_hit.max())
            raymarcher_out['relight'] = prob_hit.new_zeros(prob_hit.size(0), 3)
            raymarcher_out['hdr'] = prob_hit.new_zeros(prob_hit.size(0), 3)
            raymarcher_out['albedo'] = prob_hit.new_zeros(prob_hit.size(0), 3)
            raymarcher_out['rough'] = prob_hit.new_zeros(prob_hit.size(0), 1)
            raymarcher_out['uv_residual'] = prob_hit.new_zeros(prob_hit.size(0), 2)
            raymarcher_out['sampled_normals'] = prob_hit.new_zeros(prob_hit.size(0), 3)
            raymarcher_out['sampled_visibility'] = prob_hit.new_zeros(prob_hit.size(0), 512)

            if rel_hit.max() > 0:
                # rel_depth_min = depth[rel_hit].unsqueeze(1) - 0.06
                # rel_depth_max = depth[rel_hit].unsqueeze(1) + 0.06
                rel_depth_min = depth_min[rel_hit] - 0.03
                rel_depth_max = depth_max[rel_hit] + 0.03

                rel_raymarcher_out = self._mesh_based_ray_marcher_inverse_rendering_chunk(
                        rays_start[rel_hit],
                        rays_dir[rel_hit],
                        rel_depth_min,
                        rel_depth_max,
                        max_num_steps=relight_S,
                        triangles=triangles,
                        face_weights=face_weights,
                        texture_map=texture_map,
                        glob_rotation=glob_rotation,
                        glob_translation=glob_translation,
                        joints_RT=joints_RT,
                        **kwargs,
                )
                # relight_mask = rel_raymarcher_out['prob_hit'].unsqueeze(1)

                # raymarcher_out['relight'] = raymarcher_out['rgb'].detach().clone()
                raymarcher_out['relight'][rel_hit] = rel_raymarcher_out['rgb']
                # (relight_mask * rel_raymarcher_out['rgb']
                #     + (1 - relight_mask) * raymarcher_out['relight'][rel_hit]
                # )
                raymarcher_out['hdr'][rel_hit] = rel_raymarcher_out['hdr']
                raymarcher_out['albedo'][rel_hit] = rel_raymarcher_out['albedo']
                raymarcher_out['rough'][rel_hit] = rel_raymarcher_out['rough']
                raymarcher_out['uv_residual'][rel_hit] = rel_raymarcher_out['uv_residual']
                raymarcher_out['sampled_normals'][rel_hit] = rel_raymarcher_out['sampled_normals']
                raymarcher_out['sampled_visibility'][rel_hit] = rel_raymarcher_out['sampled_visibility']
                raymarcher_out['sampled_uv_residual'] = rel_raymarcher_out['sampled_uv_residual']

            else:
                raymarcher_out['sampled_uv_residual'] = rel_hit.new_zeros(rel_hit.size(0), relight_S + 1, 2)

        return raymarcher_out


    def _mesh_based_ray_marcher_inverse_rendering_chunk(self,
                                                        rays_start,
                                                        rays_dir,
                                                        depth_min,
                                                        depth_max,
                                                        max_num_steps,
                                                        triangles,
                                                        face_weights,
                                                        texture_map,
                                                        glob_rotation,
                                                        glob_translation,
                                                        joints_RT,
                                                        uv_normal_map,
                                                        uv_visibility_map):
        """This function performs a ray marching with inverse rendering on a
        neural field guided by an explicit mesh. The ray is defined in the
        following manner:

        rays_start (3D) x----------------------------> rays_dir (3D)
                        |                            |
                    depth_min                   depth_max

        Points will be sampled along the ray, the distance between each points
        is `(depth_max - depth_min) / max_num_steps`. When a sampled point
        is too far from the vertices of the mesh (> `self.min_dis_eps`), it
        will be skipped, assuming zero density. In total, up to
        `S = max_num_steps + 1` points will be sampled along each ray.

        # Arguments
            rays_start, rays_dir: tensor with shape (num_rays, 3)
            depth_min, depth_max: tensors with shape (num_rays, 1)
            triangles: shape (F, 3, 3)
            face_weights: shape (F, 3, 24)
            texture_map: shape (B=1, f_dim, feat_h, feat_w)
            glob_rotation: shape (3, 3)
            glob_translation: shape (1, 3)
            joints_RT: shape (J=24, 4, 4)
            density_only: boolean, skips color if True
            relight: boolean, if performing inverse rendering
            uv_normal_map: (batch=1, 3, H, W)
            uv_visibility_map: (batch=1, num_lights, H, W)

        # Returns
        """
        S = max_num_steps + 1
        N = rays_start.shape[0]
        sigma_agg = rays_start.new_zeros(size=(N, S))
        albedo_agg = rays_start.new_zeros(size=(N, S, 3))
        rough_agg = rays_start.new_zeros(size=(N, S, 1))
        uv_residual_agg = rays_start.new_zeros(size=(N, S, 2))
        hdr_agg = rays_start.new_zeros(size=(N, S, 3))
        sampled_normals_agg = rays_start.new_zeros(size=(N, S, 3))
        num_lights = uv_visibility_map.size(1)
        sampled_visibility_agg = rays_start.new_zeros(size=(N, S, num_lights))

        rays_mask = torch.ones((N,), dtype=torch.bool, device=rays_start.device)
        step_check = 4

        # Compute the depth of each sampled point in each ray
        depth_step = torch.tile(((depth_max - depth_min) / max_num_steps)[:, None], (1, S, 1))
        if self.training:
            depth_step = torch.clamp(depth_step + torch.zeros_like(depth_step).normal_(0, 1e-3), 1e-5, None)
        # accumulate steps and add depth offset (start of the ray)
        sampled_depth = torch.cumsum(depth_step, dim=1) + depth_min.unsqueeze(1)

        def _aggregate_samples_relight(step):
            nonlocal rays_mask

            sigmap = torch.relu(sigma_agg[:, :step + 1])

            # Set density for points where the normal is not facing the camera to zero
            # norms = sampled_normals_agg[:, :step + 1]
            # dotp = torch.einsum("nsd,nd->ns", norms, rays_dir)
            # sigmap = dotp.lt(0).detach().float() * sigmap # TODO back lit Normals

            free_energy = sigmap * depth_step[:, :step + 1].squeeze(2)
            shifted_free_energy = torch.cat([
                    free_energy.new_zeros(N, 1),
                    free_energy[:, :-1]
                ], dim=-1)  # shift one step, (num_rays, num_steps+1)
            a = 1 - torch.exp(-free_energy.float()) # probability of it is not empty here (N, S') -> S' steps so far
            b = torch.exp(-torch.cumsum(shifted_free_energy.float(), dim=-1)) # probability of everything is empty up to now (N, S')
            probs = (a * b).type_as(free_energy) # (N, S')
            prob_hit = probs.sum(-1) # Chance that the ray did intersect something dense (N,)
            rays_mask *= prob_hit.lt(0.999)

            albedo = (albedo_agg[:, :step + 1] * probs.unsqueeze(-1)).sum(1)
            rough = (rough_agg[:, :step + 1] * probs.unsqueeze(-1)).sum(1)
            hdr = (hdr_agg[:, :step + 1] * probs.unsqueeze(-1)).sum(1)

            return prob_hit, albedo, rough, hdr, probs
    
        for step in range(S):
            sampled_xyz = ray(rays_start, rays_dir, sampled_depth[:, step])
            if self.training:
                sampled_xyz += torch.zeros_like(sampled_xyz).normal_(0, 2e-4)

            intersec_data = self.intersect_points_with_mesh(sampled_xyz,
                    triangles, texture_map, face_weights, glob_rotation,
                    glob_translation, joints_RT,
                    uv_normal_map=uv_normal_map,
                    uv_visibility_map=uv_visibility_map)

            # Discard points that are too far from the mesh
            sampled_pts_mask = torch.abs(intersec_data['min_dis']) < self.min_dis_eps
            sampled_pts_mask *= rays_mask # check the general ray mask
            sampled_dir = rays_dir[sampled_pts_mask]

            # sampled_xyz_prime = sampled_xyz[sampled_pts_mask]

            if sampled_dir.size(0) == 0:
                # print ('DEBUG>> sampled_dir.size(0) = 0, rays_mask.shape, rays_start.shape', rays_mask.shape, rays_start.shape)
                if ((step + 1) % step_check == 0) or (step == S - 1) or (rays_mask.max() == 0):
                    prob_hit, albedo, rough, hdr, probs = _aggregate_samples_relight(step)
                if rays_mask.max() == 0:
                    break # we are done with all the rays
                continue # we still have valid non-hit rays running out
           
            _sigma, _uv_residual, _albedo, _rough, _hdr, _normals, _visibility = \
                self._ray_marcher_inner_inverse_rendering(
                    # sampled_xyz_prime,
                    sampled_dir,
                    # intersec_data['sampled_normals'][sampled_pts_mask],
                    # intersec_data['sampled_visibility'][sampled_pts_mask],
                    intersec_data['local_coords'][sampled_pts_mask],
                    intersec_data['sampled_tex'][sampled_pts_mask],
                    intersec_data['canonical_coords'][sampled_pts_mask],
                    self.envmap_dir_wld,
                    grids=intersec_data['grids'][sampled_pts_mask],
                    uv_normal_map=uv_normal_map,
                    uv_visibility_map=uv_visibility_map)

            sigma_buf = rays_start.new_zeros(size=(N,))
            sigma_buf[sampled_pts_mask] = _sigma
            sigma_agg[:, step] = sigma_buf

            albedo_buf = rays_start.new_zeros(size=(N, 3))
            albedo_buf[sampled_pts_mask] = _albedo
            albedo_agg[:, step] = albedo_buf

            rough_buf = rays_start.new_zeros(size=(N, 1))
            rough_buf[sampled_pts_mask] = _rough
            rough_agg[:, step] = rough_buf

            uv_residual_buf = rays_start.new_zeros(size=(N, 2))
            uv_residual_buf[sampled_pts_mask] = _uv_residual
            uv_residual_agg[:, step] = uv_residual_buf

            hdr_buf = rays_start.new_zeros(size=(N, 3))
            hdr_buf[sampled_pts_mask] = _hdr
            hdr_agg[:, step] = hdr_buf

            normals_buf = rays_start.new_zeros(size=(N, 3))
            normals_buf[sampled_pts_mask] = _normals
            sampled_normals_agg[:, step] = normals_buf

            visibility_buf = rays_start.new_zeros(size=(N, num_lights))
            visibility_buf[sampled_pts_mask] = _visibility
            sampled_visibility_agg[:, step] = visibility_buf

            # In every 4 iterations (steps) or in the last one
            if ((step + 1) % step_check == 0) or (step == S - 1):
                # Check whether we have a high probability of hitting something so far.
                # If so, then disable the ray by masking it out.
                prob_hit, albedo, rough, hdr, probs = _aggregate_samples_relight(step)

        # Colorspace transform
        rgb = func_linear2srgb(hdr)

        # Compute resulting normals and visibility in camera space
        not_missed = prob_hit.gt(0.001).unsqueeze(1).float() # (num_rays, 1)
        if probs.size(1) < S:
            probs = torch.cat([probs, probs.new_zeros(size=(probs.size(0), S - probs.size(1)))], dim=1)
        probs_nm = (probs / probs.sum(-1, keepdim=True).clamp(1e-3)) # (num_rays, S)
        sampled_normals = (sampled_normals_agg * (not_missed * probs_nm).unsqueeze(2)).sum(1)
        sampled_visibility = (sampled_visibility_agg * (not_missed * probs_nm).unsqueeze(2)).sum(1)
        uv_residual = (uv_residual_agg * (not_missed * probs_nm).unsqueeze(2)).sum(1)

        return {
            'prob_hit': prob_hit, # (N,)
            'rgb': rgb, # (N, 3)
            'hdr': hdr,
            'sigma': sigma_agg, # (N, S)
            'albedo': albedo, # (N, 3)
            'rough': rough, # (N, 1)
            'sampled_normals': sampled_normals, # (N, 3)
            'sampled_visibility': sampled_visibility, # (N, num_lights)
            'uv_residual': uv_residual, # (N, 2)
            'sampled_uv_residual': uv_residual_agg, # (N, S, 2)
        }


    def mesh_based_ray_marcher(self,
                               rays_start,
                               rays_dir,
                               depth_min,
                               depth_max,
                               max_num_steps,
                               density_only=False,
                               relight=False,
                               **kwargs):
        """This is a wrapper function to `_mesh_based_ray_marcher_chunk`.
        Due to memory limitations, a maximum number of rays can be processed
        at a time, Therefore, this function checks the number of rays to be
        processed, and split them into chunks. The maximum number of rays to
        be processed is set in the model in `self.max_num_rays_intersecting`.

        # Arguments
            rays_start: (num_rays, 3)
            rays_dir: (num_rays, 3)
            depth_min: (num_rays, 1)
            depth_max: (num_rays, 1)
            max_num_steps: integer
            density_only: boolean
            relight: boolean
            kwargs: dictionary with additional inputs.
                Check detailed description in `_mesh_based_ray_marcher_chunk`.

        # Returns
            The aggregated outputs from `_mesh_based_ray_marcher_chunk`.
        """

        raymarcher_out = None
        num_rays = rays_start.size(0)
        if num_rays > 0:
            max_rays = self.max_num_rays_intersecting
            num_iter = (num_rays - 1) // max_rays + 1

            if num_iter > 1:
                for iter in range(num_iter):
                    start_idx = iter * max_rays
                    end_idx = min(start_idx + max_rays, num_rays)

                    _raymarcher_out = self._mesh_based_ray_marcher_chunk(
                            rays_start=rays_start[start_idx:end_idx],
                            rays_dir=rays_dir[start_idx:end_idx],
                            depth_min=depth_min[start_idx:end_idx],
                            depth_max=depth_max[start_idx:end_idx],
                            max_num_steps=max_num_steps,
                            density_only=density_only,
                            relight=relight,
                            **kwargs)

                    if not raymarcher_out: # initialize it
                        raymarcher_out = {}
                        for k in _raymarcher_out.keys():
                            raymarcher_out[k] = [_raymarcher_out[k]] if _raymarcher_out[k] is not None else None
                    else:
                        for k in _raymarcher_out.keys():
                            if raymarcher_out[k] is not None:
                                raymarcher_out[k].append(_raymarcher_out[k])

                for k in raymarcher_out.keys():
                    if raymarcher_out[k] is not None:
                        raymarcher_out[k] = torch.cat(raymarcher_out[k], dim=0)

            else:
                raymarcher_out = self._mesh_based_ray_marcher_chunk(
                        rays_start=rays_start,
                        rays_dir=rays_dir,
                        depth_min=depth_min,
                        depth_max=depth_max,
                        max_num_steps=max_num_steps,
                        density_only=density_only,
                        relight=relight,
                        **kwargs)

        return raymarcher_out


    def _ray_marcher_inner_radiance_field(self, sampled_dir, local_coords,
                                          sampled_tex, canonical_coords,
                                          density_only=False, **kwargs):
        """
        # Arguments
            sampled_dir: shape (N, 3)
            local_coords: shape (N, 3)
            sampled_tex: shape (N, f_dim)
            canonical_coords: shape (N, 3)
            density_only: boolean, skips color if True

        # Returns
            Sigma tensor (N,) and RGB tensor (N, 3)
        """
        sigma_agg = []
        rgb_agg = []
        residual_agg = []

        # Here we split the network calls to avoid OOM errors
        max_n = self.max_num_pts_processing
        N = sampled_dir.shape[0]
        num_iter = (N - 1) // max_n + 1
        for iter in range(num_iter):
            start_idx = iter * max_n
            end_idx = min(start_idx + max_n, N)

            local_pos_features, local_tex_features, residual_deform = self.forward_deformation_net(
                local_coords[start_idx:end_idx], sampled_tex[start_idx:end_idx])

            canonical_coords_def = canonical_coords[start_idx:end_idx] + residual_deform
            if self.clip_canonical:
                canonical_coords_def = torch.clamp(canonical_coords_def, -1, 1)

            implicit_field_features, sigma = self.forward_density_net(canonical_coords_def)
            sigma_agg.append(sigma)
            residual_agg.append(residual_deform)

            if not density_only:
                rgb = self.forward_texture_net(sampled_dir[start_idx:end_idx],
                    local_tex_features, implicit_field_features)
                rgb_agg.append(rgb)

        sigma_agg = torch.cat(sigma_agg, dim=0)
        if not density_only:
            rgb_agg = torch.cat(rgb_agg, dim=0)
        residual_agg = torch.cat(residual_agg, dim=0)

        return sigma_agg, rgb_agg, residual_agg


    def _ray_marcher_inner_inverse_rendering(self,
                                            #  sampled_xyz,
                                             sampled_dir,
                                            #  sampled_normals,
                                            #  sampled_visibility,
                                             local_coords,
                                             sampled_tex,
                                             canonical_coords,
                                             lights_dir,
                                             uv_normal_map,
                                             uv_visibility_map,
                                             **kwargs):
        """
        # Arguments
            sampled_dir: shape (N, 3)
            sampled_normals: shape (N, 3)
            sampled_visibility: shape (N, L=env_h * env_w)
            local_coords: shape (N, 3)
            sampled_tex: shape (N, f_dim)
            canonical_coords: (N, 3)
            lights_dir: shape (env_h * env_w, 3)
            kwargs: commutable and possibly exclusive inputs, as:
                sampled_albedo: shape (N, f_dim)
                grids: (N, 2)

        # Returns
            TODO
        """
        # num_rays, steps = sampled_dir.shape
        sigma_agg = []
        uv_residual_agg = []
        albedo_agg = []
        rough_agg = []
        hdr_agg = []
        normals_agg = []
        visibility_agg = []

        # Here we split the network calls to avoid OOM errors
        max_n = self.max_num_pts_processing
        N = sampled_dir.shape[0]
        num_iter = (N - 1) // max_n + 1
        for iter in range(num_iter):
            start_idx = iter * max_n
            end_idx = min(start_idx + max_n, N)

            local_pos_features, local_tex_features, residual_deform = self.forward_deformation_net(
                local_coords[start_idx:end_idx], sampled_tex[start_idx:end_idx])

            canonical_coords_def = canonical_coords[start_idx:end_idx] + residual_deform
            if self.clip_canonical:
                canonical_coords_def = torch.clamp(canonical_coords_def, -1, 1)

            implicit_field_features, sigma = self.forward_density_net(canonical_coords_def)

            sampled_grids = kwargs['grids'][start_idx:end_idx]
            if self.training:
                sampled_grids += torch.zeros_like(sampled_grids).normal_(0, 1e-3)

            sampled_normals, sampled_visibility = self.forward_normals_visibility(
                sampled_grids, uv_normal_map, uv_visibility_map)

            if self.static_albedo:
                albedo, rough, uv_residual = self.forward_static_albedo_brdf(
                    local_coords[start_idx:end_idx], canonical_coords_def,
                    # local_pos_features, implicit_field_features,
                    sampled_tex[start_idx:end_idx], sampled_grids)
            else:
                albedo, rough = self.forward_brdf_net(
                    local_pos_features, implicit_field_features, kwargs['sampled_albedo'][start_idx:end_idx])

            lights_dir_chunk = lights_dir.unsqueeze(0).tile(end_idx - start_idx, 1, 1)
            # normals_chunk = sampled_normals[start_idx:end_idx]
            # visibility_chunk = sampled_visibility[start_idx:end_idx]
            # brdf, pts2l, pts2c, pts2n = \
            brdf = self.forward_brdf_model(-sampled_dir[start_idx:end_idx], lights_dir_chunk, # TODO rename this
                                         albedo, rough, sampled_normals) # (N, L, 3)

            ## DEBUG Microfact model
            # ptsxyz = sampled_xyz[start_idx:end_idx]

            # ## TODO HERE: save OBJ files with ptsxyz and the corresponting -> lights -> camera -> normal
            # save_points_lines_obj(f'dbg_pts2light.obj', ptsxyz, pts2l[:, 0], pts_color=[1, 1, 1], dir_color=[0, 0, 1])
            # save_points_lines_obj(f'dbg_pts2camera.obj', ptsxyz, 4 * pts2c, pts_color=[1, 1, 1], dir_color=[1, 0, 0])
            # save_points_lines_obj(f'dbg_pts2normals.obj', ptsxyz, 0.1 * pts2n, pts_color=[1, 1, 1], dir_color=[0, 1, 0])

            hdr = self._render(lights_dir_chunk, sampled_normals, sampled_visibility, brdf)

            sigma_agg.append(sigma)
            uv_residual_agg.append(uv_residual)
            albedo_agg.append(albedo)
            rough_agg.append(rough)
            hdr_agg.append(hdr)
            normals_agg.append(sampled_normals)
            visibility_agg.append(sampled_visibility)

        sigma_agg = torch.cat(sigma_agg, dim=0)
        uv_residual_agg = torch.cat(uv_residual_agg, dim=0)
        albedo_agg = torch.cat(albedo_agg, dim=0)
        rough_agg = torch.cat(rough_agg, dim=0)
        hdr_agg = torch.cat(hdr_agg, dim=0)
        normals_agg = torch.cat(normals_agg, dim=0)
        visibility_agg = torch.cat(visibility_agg, dim=0)

        return sigma_agg, uv_residual_agg, albedo_agg, rough_agg, hdr_agg, normals_agg, visibility_agg


    def _render(self, lights_dir, normals, visibility, brdf):
        """
        # Arguments
            lights_dir: shape (N, L=env_h * env_w, 3)
            normals: shape (N, 3)
            visibility: shape (N, L)
            brdf: shape (N, L, 3)
        """
        lcos = torch.einsum('ijk,ik->ij', lights_dir, normals) # (N, L)
        areas = self.envmap_area.reshape(1, -1, 1) # (1, L, 1)
        front_lit = lcos > 0
        lvis = front_lit * visibility # (N, L)

        def integrate(light):
            """
            # Arguments
                light: shape (L, 3)
            """
            light = lvis[:, :, None] * light[None, :, :] # NxLx3
            tmp = light * lcos[:, :, None] * areas
            light_pix_contrib = brdf * tmp #light * lcos[:, :, None] * areas # NxLx3
            hdr = torch.sum(light_pix_contrib, dim=1) #Nx3
            hdr = torch.clip(hdr, 0., 1.)
            
            return hdr # (N, 3)

        hdr = integrate(self.envmap_light)

        return hdr # (N, 3)


    def forward_deformation_net(self, local_coords, sampled_tex):
        """Forward in deformation network.

        # Arguments:
            local_coords: tensor (N, 3)
            sampled_tex: tensor (N, f_dim)
        """
        local_coords_pe = self.pe_local_fn(local_coords)
        local_pos_features = self.merge_joint_field(local_coords_pe)
        local_tex_features = torch.cat([local_pos_features, sampled_tex], dim=1)
        residual_deform = self.skinning_deform(local_tex_features)

        return local_pos_features, local_tex_features, residual_deform


    def forward_density_net(self, canonical_coords):
        """Forward the density network.
        
        # Arguments:
                canonical_coords: tensor (N, 3)
        """
        canonical_coords_pe = self.pe_canonical_fn(canonical_coords)
        implicit_field_features = self.feature_field(canonical_coords_pe)
        sigma, _ = self.predictor(implicit_field_features)
        if self.training:
            noise = torch.zeros_like(sigma).normal_(0, 1)
            sigma += noise
        # sigma = torch.relu(sigma)

        return implicit_field_features, sigma


    def forward_texture_net(self, sampled_dir, local_tex_features, implicit_field_features):
        """Forward the texture (RGB color) network.
        
        # Arguments:
            sampled_dir: tensor (N, 3)
            local_tex_features: tensor (N, tex_f_dim)
            implicit_field_features: tensor (N, implicit_f_dim)

        # Returns
            Tensor with shape (N, 3)
        """
        sampled_dir_pe = self.pe_ray_fn(sampled_dir)
        # x = torch.cat([sampled_dir_pe, local_tex_features, implicit_field_features], dim=1)
        x = torch.cat([implicit_field_features, local_tex_features, sampled_dir_pe], dim=1)
        rgb = self.renderer(x)
        # rgb = rgb / 2 + 0.5 # NEURA Neural Actor TODO
        # rgb = torch.clamp(rgb, 0, 1)
        rgb = torch.sigmoid(rgb)

        return rgb


    def forward_brdf_net(self, local_pos_features, implicit_field_features, sampled_albedo):
        import math
        """Forward the BRDF network.

        # Arguments:
            canonical_coords: tensor (N, 3)
            implicit_field_features: tensor (N, implicit_f_dim)

        # Returns
            Albedo tensor with shape (N, 3) and BRDF tensor with shape (N, brdf_dim)
        """
        x = torch.cat([local_pos_features, implicit_field_features, sampled_albedo], dim=1)
        x = self.relight_renderer(x)
        albedo = self.albedo_scale * torch.sigmoid(x[..., 0:3]) + self.albedo_bias
        brdf = torch.sigmoid(x[..., 3:]) 

        return albedo, brdf


    def forward_static_albedo_brdf(self, local_coords, canonical_coords, sampled_tex_features, grids):
        """Forward the static Albedo/BRDF UV sampling.

        # Arguments:
            local_coords: tensor (N, 3)
            canonical_coords: tensor (N, 3)
            sampled_tex_features: tensor (N, tex_f_dim)
            grids: tensor (N, 2)

        # Returns
            Albedo tensor with shape (N, 3) and BRDF tensor with shape (N, brdf_dim)
        """
        local_coords_pe = self.pe_local_fn(local_coords)
        canonical_coords_pe = self.pe_canonical_fn(canonical_coords)
        grids = grids.unsqueeze(1).unsqueeze(0) # (N, 2) -> (1, N, 1, 2)

        # x = torch.cat([local_coords_pe, canonical_coords_pe, sampled_tex_features], dim=1)
        x = torch.cat([local_coords_pe, sampled_tex_features], dim=1)
        uv_residual = self.uv_deform(x)
        uv_residual = 0.1 * (torch.sigmoid(uv_residual) - 0.5) # Bound deformation from -5%/+5%
        uv_residual = uv_residual.unsqueeze(1).unsqueeze(0) # (N, 2) -> (1, N, 1, 2)

        # First sample from the UV deformation mask
        uv_deform_mask = F.grid_sample(self.uv_deform_mask, grids, mode='nearest', align_corners=False)  # (1, 1, N, 1)
        uv_residual = uv_deform_mask.permute(0, 2, 3, 1) * uv_residual
        deformed_grids = torch.clamp(grids + uv_residual, -1, 1)

        uv_map = torch.cat([self.static_albedo_map, self.static_brdf_map], dim=1)
        brdf = F.grid_sample(uv_map, deformed_grids, mode='bilinear', align_corners=False)  # (1, 3+brdf_dim, N, 1)
        brdf = brdf.permute(0, 2, 3, 1).reshape(-1, brdf.size(1))  # (N, 3+brdf_dim)
        albedo, rough = brdf[:, :3], brdf[:, 3:]

        # albedo = F.grid_sample(self.static_albedo_map, deformed_grids, mode='bilinear', align_corners=False)  # (1, 3, N, 1)
        # albedo = albedo.permute(0, 2, 3, 1).reshape(-1, 3)  # (N, 3)
        # # albedo = self.albedo_scale * F.sigmoid(albedo) + self.albedo_bias

        # brdf = F.grid_sample(self.static_brdf_map, deformed_grids, mode='bilinear', align_corners=False)  # (1, brdf_dim, N, 1)
        # brdf_dim = brdf.size(1)
        # brdf = brdf.permute(0, 2, 3, 1).reshape(-1, brdf_dim)  # (N, brdf_dim)
        # brdf = F.sigmoid(brdf)

        return albedo, rough, uv_residual.squeeze(2).squeeze(0)


    def forward_normals_visibility(self, grids, uv_normal_map, uv_visibility_map):
        deformed_grids = grids.unsqueeze(1).unsqueeze(0) # (N, 2) -> (1, N, 1, 2)

        normals = F.grid_sample(uv_normal_map, deformed_grids, mode='bilinear', align_corners=False)  # (1, 3, N, 1)
        normals = normals.permute(0, 2, 3, 1).reshape(-1, 3)  # (N, 3)
        normals = normalize_fn(normals, axis=-1)[0]

        num_lights = uv_visibility_map.size(1)
        visibility = F.grid_sample(uv_visibility_map, deformed_grids, mode='bilinear', align_corners=False)  # (1, L, N, 1)
        visibility = visibility.permute(0, 2, 3, 1).reshape(-1, num_lights)  # (N, 3)

        return normals, visibility
        


    def forward_brdf_model(self, camera_dir, lights_dir, albedo, brdf, normal):
                        #  local_tex_features, implicit_field_features):
        """Forward the BRDF network.

        # Arguments:
            albedo: tensor (N, 3)
            normal: tensor (N, 3)
            camera_dir: tensor (N, 3)
            lights_dir: tensor (N, L=env_h * env_w, 3)
            # local_tex_features: tensor (N, tex_f_dim)
            # implicit_field_features: tensor (N, implicit_f_dim)

        # Returns
            Tensor with shape (N, L, 3)
        """
        # x = torch.cat([local_tex_features, implicit_field_features], dim=1)
        # brdf_rough = self.brdf_head_block(x)
        # brdf_rough = torch.sigmoid(brdf_rough)


        # pts2l = torch.nn.functional.normalize(lights_dir, p=2, dim=-1, eps=1e-7)
        # pts2c = torch.nn.functional.normalize(camera_dir, p=2, dim=-1, eps=1e-7)
        # pts2n = torch.nn.functional.normalize(normal, p=2, dim=-1, eps=1e-7)

        brdf = self.microfacet(pts2l=lights_dir, pts2c=camera_dir,
                               normal=normal, albedo=albedo, rough=brdf)

        return brdf #, pts2l, pts2c, pts2n

    @torch.no_grad()
    def _copy_layer_recursive(self, obj, layers, value, requires_grad=True):
        if len(layers) > 1:
            self._copy_layer_recursive(getattr(obj, layers[0]), layers[1:], value, requires_grad)
        else:
            getattr(obj, layers[0]).copy_(value)
            if requires_grad == False:
                getattr(obj, layers[0]).requires_grad_(requires_grad)


    @torch.no_grad()
    def load_geometry_weights(self, filename, requires_grad=True):
        print (f'Loading geometry weights from {filename}')
        state_dict = torch.load(filename)
        for layer in state_dict.keys():
            layers = layer.split('.')
            if layers[0] in self.geometry_layers:
                self._copy_layer_recursive(self, layers, state_dict[layer], requires_grad)

    @torch.no_grad()
    def load_radiance_weights(self, filename, requires_grad=True):
        # TODO replace filename by state_dict as input
        print (f'Loading radiance weights from {filename}')
        state_dict = torch.load(filename)
        for layer in state_dict.keys():
            layers = layer.split('.')
            if layers[0] in self.radiance_layers:
                self._copy_layer_recursive(self, layers, state_dict[layer], requires_grad)

    @torch.no_grad()
    def load_relight_weights(self, filename, requires_grad=True):
        # TODO replace filename by state_dict as input
        print (f'Loading relight weights from {filename}')
        state_dict = torch.load(filename)
        for layer in state_dict.keys():
            layers = layer.split('.')
            if layers[0] in self.relight_layers:
                self._copy_layer_recursive(self, layers, state_dict[layer], requires_grad)
            ## FIXME this is for compatibility with old models
            elif layers[0] == 'static_albedo_map':
                self._copy_layer_recursive(self, ['_static_albedo_map'], state_dict[layer], requires_grad)
            elif layers[0] == 'static_brdf_map':
                self._copy_layer_recursive(self, ['_static_brdf_map'], state_dict[layer], requires_grad)

    @torch.no_grad()
    def update_uvmap_geometry(self, grids, normals, rgb, surface_xyz, tex_size, env_map_size, inputs, pose_data,
                              num_repeat_uv_pix=1, num_steps=16, compute_visibility=False):
        """
        # Arguments
            grids: tensor with UV coordinates with shape (N, 2)
            normals: tensor with normal values with shape (N, 3)
            rgb: tensor with RGB values with shape (N, 3), or None
            surface_xyz: tensor with surface values with shape (N, 3)
            tex_size: tuple with tex map size as (tex_w, tex_h)
            env_map_size: tuple with environment map size as (lng=env_w, lat=env_h)
            inputs: dictionary with
                uv_normals: tensor that buffers the geometry normals, shape (tex_h * tex_w, 3)
                uv_tex: tensor that buffers the texture map, shape (tex_h * tex_w, 3)
                uv_tex_cnt: tensor that buffers the texture map pixel count, shape (tex_h * tex_w)
                uv_vis: tensor that buffers the visibility flags, shape (tex_h * tex_w // 4, num_env_pix=prod(env_map_size))
                uv_acc: tensor that accumulates processed points in uv space, shape (tex_h * tex_w // 4)
            pose_data: dictionary with
                'vertices':
                'texture_map':
                'glob_rotation':
                'glob_translation':
                'joints_RT':
        """
        tex_w, tex_h = tex_size
        env_w, env_h = env_map_size
        num_env_pix = env_w * env_h

        idx_normals = uv_grid_to_uv_idx(grids, tex_w, tex_h)
        idx_normals_s, idx_normals_s_ids = torch.sort(idx_normals)
        idx_normals_s_mask = torch.concat([torch.ones_like(idx_normals_s[:1]),
                                       idx_normals_s[1:] > idx_normals_s[0:-1]], dim=0).bool()

        # Sort the inputs based on the UV mapping
        normals_map_s = normals[idx_normals_s_ids]
        # rgb_map_s = rgb[idx_normals_s_ids]

        inputs['uv_normals'][idx_normals_s[idx_normals_s_mask]] += normals_map_s[idx_normals_s_mask]
        # inputs['uv_tex'][idx_normals_s[idx_normals_s_mask]] += rgb_map_s[idx_normals_s_mask]
        # inputs['uv_tex_cnt'][idx_normals_s[idx_normals_s_mask]] += 1

        if compute_visibility is False:
            return

        idx_vis = uv_grid_to_uv_idx(grids, tex_w // 2, tex_h // 2)
        idx_vis_s, idx_vis_s_ids = torch.sort(idx_vis)
        idx_vis_s_mask = torch.concat([torch.ones_like(idx_vis_s[:1]),
                                       idx_vis_s[1:] > idx_vis_s[0:-1]], dim=0).bool()

        # Sort the inputs based on the UV mapping
        normals_s = normals[idx_vis_s_ids] # (N, 3)
        surface_xyz_s = surface_xyz[idx_vis_s_ids] # (N, 3)

        # Update the mask to ignore pixels already computed (based on `uv_acc`)
        idx_vis_s_mask = inputs['uv_acc'][idx_vis_s].lt(num_repeat_uv_pix) * idx_vis_s_mask

        # Get the unique inputs based on the UV mapping
        normals_s_u = normals_s[idx_vis_s_mask] # (U, 3)
        surface_xyz_s_u = surface_xyz_s[idx_vis_s_mask] # (U, 3)

        # Prepare buffers for each point on the surface to each light source
        visibility = torch.zeros((surface_xyz_s_u.shape[0] * env_h * env_w)).to(surface_xyz_s_u.device) # (U * num_env_pix,)
        envpix_dir = torch.tile(self.envmap_dir_wld.unsqueeze(0), (surface_xyz_s_u.shape[0], 1, 1)).view(-1, 3) # (U * num_env_pix, 3)
        surface_xyz_tiled = surface_xyz_s_u.unsqueeze(1).tile(1, num_env_pix, 1).view(-1, 3) # (U * num_env_pix, 3)

        # Compute visibility only for non backlit (nbl) points
        normals_unique_tiled = normals_s_u.unsqueeze(1).tile(1, num_env_pix, 1).view(-1, 3) # (U * num_env_pix, 3)
        non_backlit_pts = torch.sum(envpix_dir * normals_unique_tiled, -1).gt(0.1) # (U * num_env_pix,)

        rays_start = surface_xyz_tiled[non_backlit_pts] # (P, 3)
        rays_dir = envpix_dir[non_backlit_pts] # (P, 3)

        pose_data['triangles'] = F.embedding(self.faces, pose_data['vertices']) # (F, 3, 3)
        pose_data['face_weights'] = F.embedding(self.faces, self.skinning_weights)  # (F, 3, 24)
        _ones = surface_xyz.new_ones(size=(rays_start.shape[0], 1))

        raymarcher_out = self.mesh_based_ray_marcher(
                            rays_start,
                            rays_dir,
                            0.01 * _ones,
                            1.0 * _ones,
                            num_steps,
                            density_only=False,
                            **pose_data)

        if raymarcher_out is not None:
            visibility[non_backlit_pts] = 1 - raymarcher_out['prob_hit'] # (P,)
            visibility = visibility.view(-1, num_env_pix) # (U, num_env_pix)
            valid_vis = visibility.mean(1).gt(0.1).float() # (U,)
            valid_norm = (normals_s_u ** 2).sum(1).gt(0.5).float() # (U,)
            vis_mask = valid_norm * valid_vis

            inputs['uv_vis'][idx_vis_s[idx_vis_s_mask]] += vis_mask.unsqueeze(1) * visibility
            inputs['uv_acc'][idx_vis_s[idx_vis_s_mask]] += vis_mask


    def _load_environment_map(self, hrd_filepath, scale=1.0, trainable=False, envmap_background_filepath=""):
        env_w, env_h = self.envmap_size

        hdr_img = cv2.imread(hrd_filepath, flags=cv2.IMREAD_ANYDEPTH)
        hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
        if hdr_img.shape[0:2] != (env_h, env_w):
            hdr_img = cv2.resize(hdr_img, (env_w, env_h), interpolation=cv2.INTER_AREA)

        envmap = np.clip(scale * np.array(hdr_img).reshape(-1, 3), 0, None)

        self.trainable_envmap = trainable
        if trainable:
            _envmap_light = torch.nn.Parameter(torch.from_numpy(envmap))
            self.register_parameter('_envmap_light', _envmap_light)
            self.relight_layers.append('_envmap_light')
        else:
            # self.register_buffer("envmap_light", torch.from_numpy(0.5 * np.ones((env_w * env_h, 3))), persistent=False)
            self.register_buffer("_envmap_light", torch.from_numpy(envmap), persistent=False)

            # init rgb factor with values around 1 (post softplus activation)
            rgb_factor = torch.nn.Parameter(np.log(np.e - 1) * torch.from_numpy(np.ones((1, 1), dtype=np.float32)))
            self.register_parameter("envmap_light_rgb_factor", rgb_factor)
            self.relight_layers.append('envmap_light_rgb_factor')

            def slow_down_rgb_factor_grad(grad):
                # return self.coef_envmap_light_rgb_factor * torch.clamp(grad, -0.1, 0.1)
                return self.coef_envmap_light_rgb_factor * grad
            self.envmap_light_rgb_factor.register_hook(slow_down_rgb_factor_grad)

        if envmap_background_filepath != "":
            hdr_img = cv2.imread(envmap_background_filepath, flags=cv2.IMREAD_ANYDEPTH)
            hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
            if hdr_img.shape[0:2] != (512, 1024):
                hdr_img = cv2.resize(hdr_img, (1024, 512), interpolation=cv2.INTER_AREA)
            hdr_img = np.clip(hdr_img, 0, None)
            self.register_buffer("envmap_background", torch.from_numpy(hdr_img), persistent=False)

            # # DEBUG:: save this EnvMap as a spherical color point cloud
            # hdri = self.envmap_background.view(-1, 3)
            # rgb = func_linear2srgb(hdri)
            # xyz, light_dir, areas = gen_light_ray_dir(512, 1024, 1000)
            # filename = f'./output/debug_envmap_background.obj'
            # save_color_pc_obj(filename, xyz, rgb)

            # # DEBUG:: convert XYZ to lat-lng again
            # pts_r_lat_lng = cart2sph(xyz)
            # new_xyz = sph2cart(pts_r_lat_lng)
            # filename = f'./output/debug_envmap_background_reverse.obj'
            # save_color_pc_obj(filename, new_xyz, rgb)


    @property
    def envmap_light(self):
        L = torch.clamp(self._envmap_light, 0, None)
        if hasattr(self, 'envmap_light_rgb_factor'):
            rgb_factor = F.softplus(self.envmap_light_rgb_factor, beta=1, threshold=6)
            L = rgb_factor * L
        return L

    @property
    def static_albedo_map(self):
        return self.albedo_scale * F.sigmoid(self._static_albedo_map) + self.albedo_bias

    @property
    def static_brdf_map(self):
        return F.sigmoid(self._static_brdf_map)

    def rotate_envmap_one_step(self):
        w, h = self.envmap_size
        tmp = self._envmap_light.reshape((h, w, 3))
        tmp = torch.cat([tmp[:, 1:], tmp[:, :1]], dim=1)
        self._envmap_light = tmp.reshape((-1, 3))

    def shift_olat_envmap(self):
        L = self._envmap_light
        self._envmap_light = torch.cat([L[-1:], L[:-1]], dim=0)

    def rotate_envmap_high_res(self, steps=4):
        self.envmap_background = torch.cat([
            self.envmap_background[:, steps:], self.envmap_background[:, :steps]
        ], dim=1)
        envmap = self.envmap_background.detach().cpu().numpy()
        envmap = self.envmap_factor * cv2.resize(envmap, self.envmap_size, interpolation=cv2.INTER_AREA)
        self._envmap_light = torch.from_numpy(envmap).to(self._envmap_light.device).reshape((-1, 3))


    def _init_albedo_brdf_maps(self, width, height, albedo_path=None, brdf_path=None):
        if albedo_path is not None:
            albedo_img = cv2.imread(albedo_path)
            albedo_img = np.flip(albedo_img, axis=0)
            albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)
            if albedo_img.shape[0:2] != (height, width):
                albedo_img = cv2.resize(albedo_img, (width, height), interpolation=cv2.INTER_AREA)
            # Safely put the data into the desired range (assuming a sigmoid activation)
            albedo_img = albedo_img / 255.9
            albedo_img = albedo_img / (1 - albedo_img)
            albedo_img /= albedo_img.max()
            postact_albedo = np.transpose(albedo_img, (2, 0, 1))[np.newaxis]
        else:
            postact_albedo = 0.5 * (self.albedo_bias + self.albedo_scale) * np.ones((1, 3, height, width), dtype=np.float32)

        if brdf_path is not None:
            raise NotImplementedError
        else:
            postact_brdf = 0.7 * np.ones((1, 1, height, width), dtype=np.float32)

        preact_albedo = func_inverse_sigmoid_np(postact_albedo)
        # preact_albedo = postact_albedo
        preact_albedo = torch.nn.Parameter(torch.from_numpy(preact_albedo.astype(np.float32)))
        self.register_parameter('_static_albedo_map', preact_albedo)
        self.relight_layers.append('_static_albedo_map')

        # def debug_grad_albedo(grad):
        #     print ('albedo', grad.shape, grad.max(), grad.mean())
        # self._static_albedo_map.register_hook(debug_grad_albedo)

        preact_brdf = func_inverse_sigmoid_np(postact_brdf)
        # preact_brdf = postact_brdf
        preact_brdf = torch.nn.Parameter(torch.from_numpy(preact_brdf.astype(np.float32)))
        self.register_parameter('_static_brdf_map', preact_brdf)
        self.relight_layers.append('_static_brdf_map')


    def set_albedo_brdf_maps(self, albedo_path=None, brdf_path=None):
        if (hasattr(self, '_static_albedo_map') is False) or ():
            raise Exception(f"Error: calling `set_albedo_brdf_maps` before `_init_albedo_brdf_maps`!")

        if albedo_path is not None:
            albedo_img = cv2.imread(albedo_path)
            albedo_img = np.flip(albedo_img, axis=0)
            albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)
            postact_albedo = np.transpose(albedo_img, (2, 0, 1))[np.newaxis]
            preact_albedo = func_inverse_sigmoid_np(postact_albedo / 255.0)
            # preact_albedo = postact_albedo / 255.
            with torch.no_grad():
                self._static_albedo_map.copy_(torch.from_numpy(preact_albedo.astype(np.float32)))

        if brdf_path is not None:
            brdf_img = cv2.imread(brdf_path)
            if brdf_img.ndim == 3:
                brdf_img = np.mean(brdf_img, axis=-1, keepdims=True)
            elif brdf_img.ndim == 2:
                brdf_img = brdf_img[..., np.newaxis]
            brdf_img = np.flip(brdf_img, axis=0)
            postact_brdf = np.transpose(brdf_img, (2, 0, 1))[np.newaxis]
            preact_brdf = func_inverse_sigmoid_np(postact_brdf / 255.0)
            # preact_brdf = postact_brdf / 255.
            with torch.no_grad():
                self._static_brdf_map.copy_(torch.from_numpy(preact_brdf.astype(np.float32)))



if __name__ == '__main__':
    from .config import ConfigContext

    with ConfigContext() as args:
        model = NeuRA()
