# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base classes for various  models.

The basic principle of differentiable rendering is two components:
    -- an field or so-called geometric field (GE)
    -- an raymarcher or so-called differentiable ray-marcher (RM)
So it can be composed as a GERM model
"""

import logging
import torch
import torch.nn as nn
import skimage.metrics
import imageio, os
import numpy as np
import copy

from tqdm import tqdm
from collections import defaultdict

from fairseq.models import BaseFairseqModel
from fairseq.utils import with_torch_seed

from fairnr.modules.encoder import get_encoder
from fairnr.modules.field import get_field
from fairnr.modules.renderer import get_renderer
from fairnr.modules.reader import get_reader
from fairnr.data.geometry import ray, compute_normal_map
from fairnr.data.geometry import fill_in
from fairnr.data.data_utils import recover_image
from fairnr.data.data_utils import recover_raw_image
from fairnr.data.data_utils import gen_light_ray_dir

logger = logging.getLogger(__name__)




_a = 0.2627
_b = 0.6780
_c = 0.0593
_d = 1.8814
_e = 1.4747

_YCbCr_to_RGB_Mat = np.array([
    [1, 1, 1],
    [0, -(_c * _d / _b), _d],
    [_e, -(_a * _e / _b), 0],
], dtype=np.float32)

def ycbcr2rgb(ycbcr):
    assert ycbcr.shape[-1] == 3, f'Expected `ycbcr.shape = (..., 3)`, got {ycbcr.shape}'
    org_shape = ycbcr.shape
    rgb = ycbcr.reshape((-1, 3)) @ _YCbCr_to_RGB_Mat
    return rgb.reshape(org_shape)








class BaseModel(BaseFairseqModel):
    """Base class"""

    ENCODER = 'abstract_encoder'
    FIELD = 'abstract_field'
    RAYMARCHER = 'abstract_renderer'
    READER = 'abstract_reader'

    def __init__(self, args, setups):
        super().__init__()
        self.args = args
        self.hierarchical = getattr(self.args, "hierarchical_sampling", False)
        
        self.reader = setups['reader']
        self.encoder = setups['encoder']
        self.field = setups['field']
        self.raymarcher = setups['raymarcher']
        self.hits_from_miss_thr = 0.1
        self.compute_visibility = False # TODO
        self.ssim_multipler = getattr(args, "ssim_multipler", 1.0)
        self.cache = None
        self._num_updates = 0
        if getattr(self.args, "use_fine_model", False):
            self.field_fine = copy.deepcopy(self.field)
        else:
            self.field_fine = None

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        reader = get_reader(cls.READER)(args)
        encoder = get_encoder(cls.ENCODER)(args)
        field = get_field(cls.FIELD)(args)
        raymarcher = get_renderer(cls.RAYMARCHER)(args)
        setups = {
            'reader': reader,
            'encoder': encoder,
            'field': field,
            'raymarcher': raymarcher
        }
        return cls(args, setups)

    @classmethod
    def add_args(cls, parser):
        get_reader(cls.READER).add_args(parser)
        get_renderer(cls.RAYMARCHER).add_args(parser)
        get_encoder(cls.ENCODER).add_args(parser)
        get_field(cls.FIELD).add_args(parser)

        # model-level args
        parser.add_argument('--hierarchical-sampling', action='store_true',
            help='if set, a second ray marching pass will be performed based on the first time probs.')
        parser.add_argument('--use-fine-model', action='store_true', 
            help='if set, we will simultaneously optimize two networks, a coarse field and a fine field.')
        parser.add_argument('--ssim-multipler', type=float, default=1.0)

    def set_num_updates(self, num_updates):
        self._num_updates = num_updates
        super().set_num_updates(num_updates)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        if (self.field_fine is None) and \
            ("field_fine" in [key.split('.')[0] for key in state_dict.keys()]):
            # load checkpoint has fine-field network, copying weights to field network
            for fine_key in [key for key in state_dict.keys() if "field_fine" in key]:
                state_dict[fine_key.replace("field_fine", "field")] = state_dict[fine_key]
                del state_dict[fine_key]

    def export_surfaces(self, **kwargs):
        encoder_states = self.preprocessing(**kwargs)
        return self.encoder.export_surfaces(self.field.forward, 50, encoder_states=encoder_states)

    @property
    def dummy_loss(self):
        return sum([p.sum() for p in self.parameters() if p.requires_grad]) * 0.0

    def forward(self, ray_split=1, **kwargs):
        # print (f'\t\tDEBUG:: call BaseModel.forward() >>>>>>>>>>>> (ray_split:', ray_split, ")")

        with with_torch_seed(self.unique_seed):   # make sure different GPU sample different rays
            ray_start, ray_dir, uv = self.reader(**kwargs) # sample the rays
        
        kwargs.update({
            'field_fn': self.field.forward,
            'input_fn': self.encoder.forward})

        if ray_split == 1:
            results = self._forward(ray_start, ray_dir, **kwargs)
        else:
            total_rays = ray_dir.shape[2]
            chunk_size = total_rays // ray_split
            results = [
                self._forward(
                    ray_start, ray_dir[:, :, i: i+chunk_size], **kwargs)
                for i in range(0, total_rays, chunk_size)
            ]
            results = self.merge_outputs(results)

        results['samples'] = {
            'sampled_uv': results.get('sampled_uv', uv),
            'ray_start': ray_start,
            'ray_dir': ray_dir
        }
        
        # caching the prediction
        self.cache = {
            w: results[w].detach() 
                if isinstance(w, torch.Tensor) 
                else results[w] 
            for w in results
        }
        return results

    def _forward(self, ray_start, ray_dir, **kwargs):
        """
        # Arguments
            ray_start: tensor with shape (batch, S=1, 1, 3)
            ray_dir: tensor with shape (batch, S=1, num_rays, 3)
            kwargs: dictionary
                'id': tensor shape (1,)
                'colors': tensor shape (batch, 1, img_h*img_w, 3)
                'intrinsics': tensor shape (batch, 1, 4, 4)
                'extrinsics': tensor shape (batch, 1, 4, 4)
                'uv': tensor shape (batch, 1, 2, img_h*img_w)
                'shape': scalar ?
                'view': tensor shape (batch, 1)
                'size': tensor shape (batch, 1, 4), with [img_h, img_w, ratio_h, ratio_w]
                'step': scalar
                'joints': tensor shape (batch, 24, 3)
                'joints_RT': tensor shape (batch, 4, 4, 24)
                'translation': tensor shape (batch, 1, 3)
                'rotation': tensor shape (batch, 3, 3)
                'pose': tensor shape (batch, 1, 72)
                'tex': tensor shape (batch, 1, 512, 512, 3)
                'field_fn': ??
                'input_fn': ??

        # Returns
        """
        # print (f'DEBUG:: call BaseModel._forward()')
        S, V, P, _ = ray_dir.size()
        assert S == 1, "we only supports single object for now."

        encoder_states = self.preprocessing(**kwargs)
        dbg_cam_pos = ray_start[0, 0, 0].detach().clone().cpu().numpy()

        # Intersect the rays with the underlying SMPL model, `hits` is a boolean vector
        ray_start, ray_dir, intersection_outputs, hits, sampled_uv = \
            self.intersecting(ray_start, ray_dir, encoder_states, **kwargs)
        
        # save the original rays
        ray_start0 = ray_start.reshape(-1, 3).clone() # (num_rays, 3)
        ray_dir0 = ray_dir.reshape(-1, 3).clone() # (num_rays, 3)

        P = ray_dir.size(1) // V
        all_results = defaultdict(lambda: None)
        if hits.sum() > 0:
            intersection_outputs = { # Select only the outputs from a hit in the mesh
                name: outs[hits] for name, outs in intersection_outputs.items()}
            ray_start, ray_dir = ray_start[hits], ray_dir[hits]
            encoder_states_nobatch = { # removes the batch dim from torch tensors (assuming batch=1 ?)
                name: s.reshape(-1, s.size(-1)) 
                    if isinstance(s, torch.Tensor) else s
                for name, s in encoder_states.items()}   
           
            samples, all_results = self.raymarching(               # ray-marching
                ray_start, ray_dir, intersection_outputs, encoder_states_nobatch)
   
            if self.hierarchical:   # hierarchical sampling
                intersection_outputs = self.prepare_hierarchical_sampling(
                    intersection_outputs, samples, all_results)
                coarse_results = all_results.copy()
                
                samples, all_results = self.raymarching(
                    ray_start, ray_dir, intersection_outputs, encoder_states_nobatch, fine=True)
                all_results['coarse'] = coarse_results

        hits = hits.reshape(-1) # (num_rays,)

        try:
            if self.compute_visibility:
                cache_depths_hits = all_results['depths'][:, None].clone()
                cache_surface_hits = all_results['missed'].lt(self.hits_from_miss_thr).clone()

            all_results = self.postprocessing(ray_start0, ray_dir0, all_results, hits, (S, V, P))

            if self.compute_visibility:
                # Compute the normals, used to speed up the visibility estimation
                img_w = int(kwargs['size'][0, 0, 1].cpu().numpy())
                normals = compute_normal_map( # (img_h*img_w, 3)
                    ray_start0[:1].float(),
                    ray_dir0.float(),
                    all_results['depths'][0, 0].float(),
                    kwargs['extrinsics'][0, 0].float().inverse(),
                    img_w, world_normals=True)
                normals = normals[hits]

                dbg_surf = None # []
                dbg_env_visibility = None # []
                num_lat = 2 # 16
                num_lng = 4 # 32
                lvis_maps = self.compute_visibility_fn(ray_start, ray_dir, cache_surface_hits,
                        cache_depths_hits, normals, encoder_states, num_lat, num_lng,
                        dbg_surf=dbg_surf,
                        dbg_env_visibility=dbg_env_visibility,
                        **kwargs)
                # We have to fill-in the values in lvis to complete all the rays in the image
                num_envpix = lvis_maps.size(-1)
                all_results['lvis'] = fill_in((S * V * P, num_envpix), hits,
                    lvis_maps, 0.0, lvis_maps.device).view(S, V, P, num_envpix)

            # Debug surface points and visibility rays
            if self.compute_visibility:
                s = 1000
                view_id = kwargs['view'].detach().clone().cpu().numpy()[0, 0]
                if dbg_surf:
                    verts = dbg_surf[0]['points']
                    # num_pts = len(verts)
                    with open(f'debug/surface_{view_id:03d}.obj', 'w') as fid:
                        # Write coordinate frame
                        fid.write(f'v {s * 0} {s * 0} {s * 0} 0.5 0.5 0.5\n')
                        fid.write(f'n 1 1 1\n')
                        fid.write(f'v {s * 1} {s * 0} {s * 0} 1.0 0.0 0.0\n')
                        fid.write(f'n 1 0 0\n')
                        fid.write(f'v {s * 0} {s * 1} {s * 0} 0.0 1.0 0.0\n')
                        fid.write(f'n 0 1 0\n')
                        fid.write(f'v {s * 0} {s * 0} {s * 1} 0.0 0.0 1.0\n')
                        fid.write(f'n 0 0 1\n')
                        fid.write(f'l 1 2\n')
                        fid.write(f'l 1 3\n')
                        fid.write(f'l 1 4\n')

                        # Write camera position
                        fid.write(f'v {s * dbg_cam_pos[0]} {s * dbg_cam_pos[1]} {s * dbg_cam_pos[2]} 1.0 0.0 1.0\n')
                        fid.write(f'n 0 0 1\n')

                        for v in verts:
                            fid.write(f'v {s * v[0]:.3f} {s * v[1]:.3f} {s * v[2]:.3f} 0.1 0.3 1.0\n')
                        # verts = dbg_surf[0]['points'] + s * 0.1 * dbg_surf[0]['normals']
                        # for v in verts:
                        #     fid.write(f'v {s * v[0]:.3f} {s * v[1]:.3f} {s * v[2]:.3f} 1.0 0.1 0.3\n')
                        # for n in range(1, num_pts + 1):
                        #     fid.write(f'l {n} {n + num_pts}\n')
                        verts_normals = dbg_surf[0]['normals']
                        for vn in verts_normals:
                            fid.write(f'n {vn[0]:.3f} {vn[1]:.3f} {vn[2]:.3f}\n')

                        if dbg_env_visibility:
                            num_env = len(dbg_env_visibility)
                            center = np.array([1, 1, 1])
                            for i in range(num_env):
                                lat_i = i // num_lng
                                lng_i = i - lat_i * num_lng
                                
                                envpix_dir = dbg_env_visibility[i]['envpix_dir']
                                for dt in np.linspace(0, s, 32):
                                    p = center + dt * envpix_dir
                                    c = np.clip(ycbcr2rgb(np.array([
                                        0.5 * dt / s + 0.25,
                                        lng_i / num_lng - 0.5,
                                        0.5 - lat_i / num_lat, # lat is up side down
                                    ])), 0, 1)
                                    fid.write(f'v {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {c[0]:.1f} {c[1]:.1f} {c[2]:.1f}\n')


                if dbg_env_visibility:
                    # valid_pts = all_results['missed'][0, 0].lt(self.hits_from_miss_thr)
                    # pts3d = all_results['xyz'][0, 0, valid_pts].cpu().numpy()
                    vertices3d = torch.matmul(encoder_states['vertices'][0], encoder_states['R'][0]) + encoder_states['T'][0]
                    vertices3d = vertices3d.cpu().numpy()
                    num_env = len(dbg_env_visibility)
                    for i in range(num_env):
                        with open(f'debug/env_{view_id:03d}_{i:03d}.obj', 'w') as fid:
                            for v in vertices3d: # SMPL vertices - dark gray
                                fid.write(f'v {s * v[0]:.3f} {s * v[1]:.3f} {s * v[2]:.3f} 0.1 0.1 0.1\n')

                            v = dbg_env_visibility[i]['light']
                            fid.write(f'v {s * v[0]:.3f} {s * v[1]:.3f} {s * v[2]:.3f} 1.0 1.0 0.0\n')
                            for v in dbg_env_visibility[i]['surf_xyz_non_backlit']:
                                fid.write(f'v {s * v[0]:.3f} {s * v[1]:.3f} {s * v[2]:.3f} 0.5 0.5 0.0\n')
                            for v in dbg_env_visibility[i]['surf_xyz_backlit']:
                                fid.write(f'v {s * v[0]:.3f} {s * v[1]:.3f} {s * v[2]:.3f} 0.0 0.3 0.1\n')
                            for v in dbg_env_visibility[i]['surf_xyz_visible']:
                                fid.write(f'v {s * v[0]:.3f} {s * v[1]:.3f} {s * v[2]:.3f} 1.0 1.0 0.7\n')

        except AttributeError as e:
            print("data problem!! please check: {}".format(kwargs["path"]))
            
            # hack raise OOM error to recover?
            raise RuntimeError("out of memory")

        if self.hierarchical:
            all_results['coarse'] = self.postprocessing(
                ray_start, ray_dir, all_results['coarse'], hits, (S, V, P))
        
        if sampled_uv is not None:
            all_results['sampled_uv'] = sampled_uv
        if hasattr(self.encoder, "compute_additional_loss"):
            all_results["additional_loss"] = self.encoder.compute_additional_loss(encoder_states)
        
        all_results['other_logs'] = self.add_other_logs(all_results)
        
        return all_results


    @torch.no_grad()
    def compute_visibility_fn(self, ray_start, ray_dir,
            surface_hits, surface_depths, surface_normals,
            encoder_states, envmap_h, envmap_w,
            dbg_surf=None,
            dbg_env_visibility=None,
            **kwargs):
        """Computes the visibility of each surface point considering the light source defined by
        an environment map and the implicit field from NeuralActor.

        # Arguments
            ray_start: tensor with shape (num_rays, 3)
            ray_dir: tensor with shape (num_rays, 3)
            surface_hits: tensor with shape (num_rays,)
            surface_depths: tensor with the depth of the surface intersection for each ray with shape (num_rays, 1)
            surface_normals: tensor with normals (x,y,z) value for each ray with shape (num_rays, 3)
            encoder_states: default dictionary with encoder states
            envmap_h, envmap_w: integers, resolution of the environment map

        # Returns
            Tensor encoding a binary visibility with shape (?)
        """
        # Here, we compute the visibility of the surface points
        points_xyz = ray_start + ray_dir * surface_depths
        S, V, P = 1, 1, surface_hits.size(0)

        points_xyz = points_xyz[surface_hits] # (num_hits, 3)
        normals = surface_normals[surface_hits] # (num_hits, 3)
        if isinstance(dbg_surf, list):
            dbg_surf.append({
                'points': points_xyz.to('cpu').numpy(),
                'normals': normals.to('cpu').numpy(),
            })

        envmap_dir_wld = gen_light_ray_dir(envmap_h, envmap_w) # (envmap_h * envmap_w, 3)
        envmap_dir_wld = torch.from_numpy(envmap_dir_wld).to(points_xyz.device) # (envmap_h * envmap_w, 3)
        lvis = []

        # for i_dir in range(envmap_h * envmap_w):
        for i_dir in tqdm(range(envmap_h * envmap_w)):
            envpix_dir = torch.tile(envmap_dir_wld[i_dir : i_dir + 1], (points_xyz.size(0), 1)) # (num_rays, 3)

            # Compute visibility only for non backlit (nbl) points
            non_backlit_pts = torch.sum(envpix_dir * normals, -1).gt(0)

            if non_backlit_pts.sum() > 0:
                points_xyz_nbl = points_xyz[non_backlit_pts]
                envpix_dir_nbl = envpix_dir[non_backlit_pts]
                surf_xyz_nbl = points_xyz_nbl + envpix_dir_nbl * 0.02 # slightly shift the starting of the ray towards the light source

                surf_xyz_nbl, envpix_dir_nbl, intersection_outputs, mesh_hits, _sampled_uv = self.intersecting(
                    surf_xyz_nbl[None, None, ...], envpix_dir_nbl[None, None, ...], encoder_states, **kwargs)

                # save the original rays
                surf_xyz0_nbl = points_xyz_nbl.reshape(-1, 3).clone() # (num_nbl_points, 3)
                envpix_dir0_nbl = envpix_dir_nbl.reshape(-1, 3).clone() # (num_nbl_points, 3)
                all_results_env = defaultdict(lambda: None)

                if mesh_hits.sum() > 0:
                    intersection_outputs = { # Select only the outputs from a hit in the mesh
                        name: outs[mesh_hits] for name, outs in intersection_outputs.items()}
                    surf_xyz_nbl, envpix_dir_nbl = surf_xyz_nbl[mesh_hits], envpix_dir_nbl[mesh_hits]

                    encoder_states_nobatch = { # removes the batch dim from torch tensors (assuming batch=1 ?)
                        name: s.reshape(-1, s.size(-1)) 
                            if isinstance(s, torch.Tensor) else s
                        for name, s in encoder_states.items()}

                    _samples, all_results_env = self.raymarching( # ray-marching
                        surf_xyz_nbl, envpix_dir_nbl, intersection_outputs, encoder_states_nobatch)

                    # For debugging -> TODO export here all the queried points and their corresponding probabilities of hit
                    # if isinstance(dbg_surf_xyz_start, list) and isinstance(dbg_surf_xyz_dir, list):
                    #     dbg_surf_xyz_start.append(surf_xyz0_nbl.to('cpu').numpy())
                    #     dbg_surf_xyz_dir.append(envpix_dir_nbl.to('cpu').numpy())

                    # define visibly envmap if the ray was missed ?? TODO check better approach to this
                    all_results_env['lvis'] = all_results_env['missed'][:, None].ge(0.0001).to(dtype=surf_xyz0_nbl.dtype)

                    if isinstance(dbg_env_visibility, list):
                        vis_hits = all_results_env['missed'].ge(0.0001)
                        dbg_env_visibility.append({
                            'light': (points_xyz + 3 * envpix_dir).mean(0).to('cpu').numpy(),
                            'envpix_dir': envpix_dir[0].to('cpu').numpy(),
                            'surf_xyz_non_backlit': surf_xyz_nbl,
                            'surf_xyz_backlit': points_xyz[torch.logical_not(non_backlit_pts)],
                            'surf_xyz_visible': surf_xyz_nbl[vis_hits],
                        })

                mesh_hits = mesh_hits.reshape(-1) # (num_hits,)

                try:
                    # First, fill-in rays that are not intersecting with the underlying mesh
                    all_results_env = self.postprocessing(surf_xyz0_nbl, envpix_dir0_nbl,
                        all_results_env, mesh_hits, (1, 1, surf_xyz0_nbl.size(0)))
                    # Remove batch and object dims added by postprocessing()
                    all_results_env = {
                        name: s[0,0] 
                            if isinstance(s, torch.Tensor) and s.ndim >= 2 else s
                            for name, s in all_results_env.items()
                    }

                    # Second, fill-in backlit rays
                    all_results_env = self.postprocessing(points_xyz, envpix_dir,
                        all_results_env, non_backlit_pts, (1, 1, points_xyz.size(0)))
                    lvis.append(all_results_env['lvis'])

                except Exception as e:
                    print("Woops, data problem in visibility!! please check: {}".format(kwargs["path"]))

            # If all rays are not intersecting the envmap, then fill in lvis with zeros
            if (non_backlit_pts.sum() == 0) or (mesh_hits.sum() == 0):
                lvis.append(torch.zeros((1, 1, non_backlit_pts.size(0), 1), dtype=torch.float32, device=non_backlit_pts.device))

        all_lvis = torch.concat(lvis, dim=-1)
        num_envpix = all_lvis.size(-1)
        all_lvis = fill_in((S, V, P, num_envpix), surface_hits, all_lvis, 0.0, ray_start.device).view(S * V * P, num_envpix)
        
        return all_lvis


    def preprocessing(self, **kwargs):
        raise NotImplementedError
    
    def postprocessing(self, ray_start, ray_dir, all_results, hits, sizes):
        raise NotImplementedError

    def intersecting(self, ray_start, ray_dir, encoder_states):
        raise NotImplementedError
    
    def raymarching(self, ray_start, ray_dir, intersection_outputs, encoder_states, fine=False):
        raise NotImplementedError

    def prepare_hierarchical_sampling(self, intersection_outputs, samples, all_results):
        raise NotImplementedError

    def add_other_logs(self, all_results):
        raise NotImplementedError

    def merge_outputs(self, outputs):
        new_output = {}
        for key in outputs[0]:
            if isinstance(outputs[0][key], torch.Tensor) and outputs[0][key].dim() > 2:
                new_output[key] = torch.cat([o[key] for o in outputs], 2)
            else:
                new_output[key] = outputs[0][key]        
        return new_output

    @torch.no_grad()
    def visualize(self, sample, output=None, shape=0, view=0, export_raw=False, **kwargs):
        """Visualize the results. If no output is given, expect to have called `forward()` first,
        so this function uses the cached predictions.

        # Arguments
            sample: dictionary with sample related data
            output: optional output (from BaseModel), or used cached results
            shape and view: sample indices ??

        # Returns
            Output images.
        """
        width = int(sample['size'][shape, view][1].item())
        img_id = '{}_{}'.format(sample['shape'][shape], sample['view'][shape, view])
        
        if output is None:
            assert self.cache is not None, "need to run forward-pass"
            output = self.cache  # make sure to run forward-pass.

        sample.update(output['samples'])
        images = {}
        images = self._visualize(images, sample, output, [img_id, shape, view, width, 'render'])
        images = self._visualize(images, sample, sample, [img_id, shape, view, width, 'target'])
        if 'coarse' in output:  # hierarchical sampling
            images = self._visualize(images, sample, output['coarse'], [img_id, shape, view, width, 'coarse'])

        if export_raw:
            images = {
            tag: recover_raw_image(images[tag]['img'], width, tag) 
                for tag in images if images[tag] is not None
            }

        else:
            images = {
                tag: recover_image(width=width, **images[tag]) 
                    for tag in images if images[tag] is not None
            }

        return images


    def _visualize(self, images, sample, output, state, **kwargs):
        img_id, shape, view, width, name = state

        if 'colors' in output and output['colors'] is not None:
            images['{}_color/{}:HWC'.format(name, img_id)] ={
                'img': output['colors'][shape, view],
                'min_val': float(self.args.min_color)
            }

        if 'missed' in output and output['missed'] is not None:
            images['{}_alpha/{}:HWC'.format(name, img_id)] = {
                # 'img': 1 - output['missed'][shape, view],
                'img': output['missed'][shape, view].lt(self.hits_from_miss_thr).to(dtype=output['missed'].dtype),
                'min_val': 0, 'max_val': 1}

        if 'depths' in output and output['depths'] is not None:
            min_depth, max_depth = output['depths'].min(), output['depths'].max()
            if getattr(self.args, "near", None) is not None:
                min_depth = 1 # self.args.near
                max_depth = 8 # self.args.far
            images['{}_depth/{}:HWC'.format(name, img_id)] = {
                'img': output['depths'][shape, view], 
                'min_val': min_depth, 
                'max_val': max_depth}
            normals = compute_normal_map(
                sample['ray_start'][shape, view].float(), # (1, 1, 1, 3)
                sample['ray_dir'][shape, view].float(), # (1, 1, img_h*img_w, 3)
                output['depths'][shape, view].float(), # (1, 1, img_h*img_w)
                sample['extrinsics'][shape, view].float().inverse(), # (1, 1, 4, 4)
                width, world_normals=True)
            images['{}_normal/{}:HWC'.format(name, img_id)] = {
                'img': normals, 'min_val': -1, 'max_val': 1}
            
            # generate point clouds from depth
            # images['{}_point/{}'.format(name, img_id)] = {
            #     'img': torch.cat(
            #         [ray(sample['ray_start'][shape, view].float(), 
            #             sample['ray_dir'][shape, view].float(),
            #             output['depths'][shape, view].unsqueeze(-1).float()),
            #          (output['colors'][shape, view] - self.args.min_color) / (1 - self.args.min_color)], 1),   # XYZRGB
            #     'raw': True }

        if 'xyz' in output and output['xyz'] is not None:
            min_xyz, max_xyz = output['xyz'].min(), output['xyz'].max()
            images['{}_xyz/{}:HWC'.format(name, img_id)] = {
                'img': output['xyz'][shape, view], 
                'min_val': min_xyz, 
                'max_val': max_xyz}

        if 'lvis' in output and output['lvis'] is not None:
            images['{}_lvis/{}:HWC'.format(name, img_id)] = {
                'img': output['lvis'][shape, view], 
                'min_val': 0, 
                'max_val': 1}

        if 'z' in output and output['z'] is not None:
            images['{}_z/{}:HWC'.format(name, img_id)] = {
                'img': output['z'][shape, view], 'min_val': 0, 'max_val': 1}
            
        # if 'normal' in output and output['normal'] is not None:
        #     images['{}_predn/{}:HWC'.format(name, img_id)] = {
        #         'img': output['normal'][shape, view], 'min_val': -1, 'max_val': 1}

        return images

    def add_eval_scores(self, logging_output, sample, output, criterion, scores=['ssim', 'psnr', 'lpips'], outdir=None):
        predicts, targets = output['colors'], sample['colors']
        no_missed = 'missed' in output
        
        ssims, psnrs, lpips, rmses = [], [], [], []
        for s in range(predicts.size(0)):
            for v in range(predicts.size(1)):
                width = int(sample['size'][s, v][1])
                t = recover_image(targets[s, v],  width=width, min_val=float(self.args.min_color))
                tn = t.numpy()
                if no_missed:
                    p = recover_image(predicts[s, v], width=width, min_val=float(self.args.min_color))
                    pn = p.numpy()
                    p, t = p.to(predicts.device), t.to(targets.device)
                    
                    if 'ssim' in scores:
                        pass
                        # ssims += [skimage.metrics.structural_similarity(pn, tn, multichannel=True, data_range=1) * self.ssim_multipler]
                    if 'psnr' in scores:
                        psnrs += [skimage.metrics.peak_signal_noise_ratio(pn, tn, data_range=1)]
                    if 'lpips' in scores: # and hasattr(criterion, 'lpips'):
                        with torch.no_grad():
                            lpips += [criterion.lpips(
                                2 * p.unsqueeze(-1).permute(3,2,0,1) - 1,
                                2 * t.unsqueeze(-1).permute(3,2,0,1) - 1).item()]

                if 'depths' in sample:
                    td = sample['depths'][sample['depths'] > 0]
                    pd = output['depths'][sample['depths'] > 0]
                    rmses += [torch.sqrt(((td - pd) ** 2).mean()).item()]

                if outdir is not None:
                    def imsave(folder, filename, image):
                        os.makedirs(f"{outdir}/{folder}", exist_ok=True)
                        imageio.imsave(f"{outdir}/{folder}/{filename}", (image * 255).astype('uint8'))
                    
                    figname = 'v{:03d}_f{:06d}.png'.format(sample['view'][s, v], sample['id'][s])

                    imsave('target', figname, tn)
                    if no_missed:
                        imsave('output', figname, pn)
                        imsave('normal', figname, recover_image(
                            compute_normal_map(
                                sample['ray_start'][s, v].float(), 
                                sample['ray_dir'][s, v].float(),
                                output['depths'][s, v].float(), 
                                sample['extrinsics'][s, v].float().inverse(), 
                                width=width),
                            min_val=-1, max_val=1, width=width).numpy())
                    
                    if 'voxel_edges' in output and output['voxel_edges'] is not None:
                        voxel_image = output['voxel_edges'][s, v].float()
                        if getattr(self.args, "no_textured_mesh", False):
                            voxel_image = recover_image(
                                img=voxel_image,
                                min_val=0, max_val=1,
                                weight=compute_normal_map(
                                    sample['ray_start'][s, v].float(), 
                                    sample['ray_dir'][s, v].float(),
                                    output['voxel_depth'][s, v].float(), 
                                    sample['extrinsics'][s, v].float().inverse(), 
                                    width=width, proj=True),
                                width=width).numpy()
                        else:
                            voxel_image = voxel_image.reshape(-1, width, 3).cpu().numpy()
                        imsave('voxel', figname, voxel_image)

                    if 'skel_edges' in output and output['skel_edges'] is not None:
                        skel_image = output['skel_edges'][s, v].float()
                        skel_image = recover_image(width=width,
                        **{
                            'img': skel_image, 
                            'min_val': 0, 
                            'max_val': 1,
                            'weight':
                                compute_normal_map(
                                    sample['ray_start'][s, v].float(), 
                                    sample['ray_dir'][s, v].float(),
                                    output['skel_depth'][s, v].float(), 
                                    sample['extrinsics'][s, v].float().inverse(), 
                                    width, proj=True)
                            }).numpy()
                        imsave('skel', figname, skel_image)

        if len(ssims) > 0:
            logging_output['ssim_loss'] = np.mean(ssims)
        if len(psnrs) > 0:
            logging_output['psnr_loss'] = np.mean(psnrs)
        if len(lpips) > 0:
            logging_output['lpips_loss'] = np.mean(lpips)
        if len(rmses) > 0:
            logging_output['rmses_loss'] = np.mean(rmses)

    def adjust(self, **kwargs):
        raise NotImplementedError

    @property
    def text(self):
        return "fairnr BaseModel"

    @property
    def unique_seed(self):
        return self._num_updates * 137 + self.args.distributed_rank
