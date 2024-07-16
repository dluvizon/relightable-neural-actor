import os
import sys
import yaml
import time
import string
import argparse

import numpy as np
from pathlib import Path


project_dir = os.path.abspath(__file__).replace('/rna/config.py','')



def decode_str_tuple(filter):
    if filter[0] == "(":
        filter = filter[1:]
    if filter[-1] == ")":
        filter = filter[:-1]
    return [int(v) for v in filter.split(',')]


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description = 'Relightable Neural Actor')
    parser.add_argument('-f', type=str, help='Default argument (used only for compatibility in Jupyter Lab)')
    parser.add_argument('--config_yaml', type=str,
                        default=os.path.join(project_dir, 'configs', 'default.yaml'),
                        help='Config YAML file') 
    parser.add_argument('--data_root', type=str, help='Path to the dataset')
    parser.add_argument('--log_dir', type=str, default="output", help='Path to the output directory')
    parser.add_argument('--experiment', type=str, help='String, experiment name')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--compute_visibility', action='store_true',
                        help='If set, compute the visibility maps at inference (very slow).')

    mesh_group = parser.add_argument_group(title='Explicit mesh')
    mesh_group.add_argument('--canonical_mesh', type=str, default="smpl/canonical.obj")
    mesh_group.add_argument('--transform_tpose', type=str, default="smpl/transform_tpose.json")
    mesh_group.add_argument('--uvmapping', type=str, default="smpl/uvmapping.obj")
    mesh_group.add_argument('--skinning_weights', type=str, default="smpl/skinning_weights.txt")
    mesh_group.add_argument('--uv_mask', type=str, default="smpl/uv_mask.png")

    data_group = parser.add_argument_group(title='Dataset options')
    data_group.add_argument('--intrinsics', type=str, default="cameras/intr/")
    data_group.add_argument('--extrinsics', type=str, default="cameras/extr/")
    data_group.add_argument('--transform', type=str, default="smpl/params")
    data_group.add_argument('--envmap_path', type=str, default="")
    data_group.add_argument('--envmap_background', type=str, default="")
    data_group.add_argument('--envmap_factor', type=float, default=1.0)
    data_group.add_argument('--train_cameras', type=str, default="0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116")
    data_group.add_argument('--valid_cameras', type=str, default="0,7,14,18,27,40")
    data_group.add_argument('--train_frames', type=str, default="0..1000")
    data_group.add_argument('--valid_frames', type=str, default="0..1")
    data_group.add_argument('--valid_res_subsample', type=int, default=4,
                            help='Image subsampling (downscaling) during validation.')
    data_group.add_argument('--rand_step_probs', type=str, default="8,7,6,5,4,3,2,1")
    data_group.add_argument('--crop_size', type=int, default=64)
    data_group.add_argument('--tex_file_ext', type=str, default='jpg')
    data_group.add_argument('--rotate_envmap', action='store_true')
    data_group.add_argument('--load_frames', action='store_true') # checked in validation or inference only
    data_group.add_argument('--image_size', type=str, default='') # tuple (w, h), must be given if load_frames=False
    data_group.add_argument('--rotating_camera', action='store_true') # When set True, ignore the {mode}_camera argument (valid only)
    data_group.add_argument('--olat', action='store_true') # When set True, shift env map (better using OLAT env map)

    model_group = parser.add_argument_group(title='Model options')
    model_group.add_argument('--model_type', type=str, default='mlp',
            help="Model type, 'mlp' or 'triplane'.")
    model_group.add_argument('--num_points_per_ray', type=int, default=64,
            help='Number of points per epoch.')
    model_group.add_argument('--min_dis_eps', type=float, default=0.06,
            help='Minimum distance between the mesh and the rays.')
    model_group.add_argument('--max_num_rays_intersecting', type=int, default=2**18,
            help='Maximum number of rays to be intersected with the mesh at once.')
    model_group.add_argument('--max_num_pts_processing', type=int, default=2**20,
            help='Maximum number of points to be processed by the implicit models at once.')
    model_group.add_argument('--geometry_weights', type=str, default='',
            help='Path to a .pth file with weights of a pre-trained radiance field model.')
    model_group.add_argument('--relight_weights', type=str, default='',
            help='Path to a .pth file with weights of a pre-trained relightable model.')
    model_group.add_argument('--normalnet_weights', type=str, default='',
            help='Path to a .pth file with weights of a pre-trained CNN for refine normal maps.')
    model_group.add_argument('--visibilitynet_weights', type=str, default='',
            help='Path to a .pth file with weights of a pre-trained CNN for refine visibility maps.')
    model_group.add_argument('--uv_deform_mask', type=str, default='',
            help='Path to an image file with a mask to where apply UV deformation.')
    model_group.add_argument('--overwrite_albedo_map', type=str, default='',
            help='Path to an image file with a new albedo map.')
    model_group.add_argument('--overwrite_brdf_map', type=str, default='',
            help='Path to an image file with a new BRDF map.')
    model_group.add_argument('--relight', action='store_true')
    model_group.add_argument('--env_map_w', type=int, default=32)
    model_group.add_argument('--env_map_h', type=int, default=16)
    model_group.add_argument('--use_uv_nets', action='store_true')
    model_group.add_argument('--use_uv_nets_norm_only', action='store_true')
    model_group.add_argument('--albedo_scale', type=float, default=0.77)
    model_group.add_argument('--albedo_bias', type=float, default=0.03)
    model_group.add_argument('--clip_canonical', action='store_true')
    model_group.add_argument('--static_albedo', action='store_true')
    model_group.add_argument('--static_albedo_res', type=int, default=1024)
    model_group.add_argument('--baseline_text_features', action='store_true')
    model_group.add_argument('--baseline_na_canonical', action='store_true')

    train_group = parser.add_argument_group(title='Training options')
    train_group.add_argument('--lr', type=float, default=0.0002)
    train_group.add_argument('--epochs', type=int, default=1000)
    train_group.add_argument('--start_epoch', type=int, default=0)
    train_group.add_argument('--batch_size', type=int, default=1)
    train_group.add_argument('--num_iter_per_epoch', type=int, default=10000,
            help='Number of iterations per epoch. Use -1 to iterate once in the dataset size.')
    train_group.add_argument('--num_sampled_triangles', type=int, default=2,
            help='Number of sampled triangles for each pose during training')
    train_group.add_argument('--num_sampled_cameras', type=int, default=1,
            help='Number of sampled cameras for each pose during training')
    train_group.add_argument('--train_full_model', action='store_true')
    train_group.add_argument('--warming_up_steps', type=int, default=1000)
    train_group.add_argument('--warming_up_factor', type=float, default=0.01)
    train_group.add_argument('--coef_loss_l2', type=float, default=10)
    train_group.add_argument('--coef_loss_vgg', type=float, default=0.01)
    train_group.add_argument('--coef_loss_mask', type=float, default=0.01)
    train_group.add_argument('--coef_loss_hard', type=float, default=0.01)
    train_group.add_argument('--coef_loss_residual', type=float, default=0.0001)
    train_group.add_argument('--coef_loss_uv_residual', type=float, default=0.001)
    train_group.add_argument('--coef_loss_albedo', type=float, default=0.01)
    train_group.add_argument('--coef_loss_rough', type=float, default=0.01)
    train_group.add_argument('--coef_uv_normals', type=float, default=10)
    train_group.add_argument('--coef_envmap_light_rgb_factor', type=float, default=0.0)
    

    parsed_args = parser.parse_args(args=input_args)

    # Update args with the content from yml file
    if parsed_args.config_yaml:
        if os.path.isfile(parsed_args.config_yaml):
            with open(parsed_args.config_yaml, 'r') as fip:
                configs_update = yaml.full_load(fip)

                for key, value in configs_update.items():
                    # use the arguments from YAML that are not given as inputs, i.e.,
                    # arguments in the input cmd have higher priority
                    appear_in_input_args = False
                    for input_arg in input_args:
                        if isinstance(input_arg, str):
                            if f'--{key}' in input_arg:
                                appear_in_input_args = True

                    if appear_in_input_args == False:
                        setattr(parsed_args, key, value)

        else:
            print (f'Warning: file {parsed_args.config_yaml} does not exist!')

    # Make relative paths to absolute paths based on data_root
    _abs_path = lambda x : x if os.path.isabs(x) else os.path.join(parsed_args.data_root, x)

    parsed_args.canonical_mesh = _abs_path(parsed_args.canonical_mesh)
    parsed_args.transform_tpose = _abs_path(parsed_args.transform_tpose)
    parsed_args.uvmapping = _abs_path(parsed_args.uvmapping)
    parsed_args.skinning_weights = _abs_path(parsed_args.skinning_weights)
    parsed_args.uv_mask = _abs_path(parsed_args.uv_mask)
    if parsed_args.uv_deform_mask != "":
        parsed_args.uv_deform_mask = _abs_path(parsed_args.uv_deform_mask)
    else:
        parsed_args.uv_deform_mask = None

    parsed_args.intrinsics = _abs_path(parsed_args.intrinsics)
    parsed_args.extrinsics = _abs_path(parsed_args.extrinsics)
    parsed_args.transform = _abs_path(parsed_args.transform)
    parsed_args.crop_size = (parsed_args.crop_size, parsed_args.crop_size)
    parsed_args.envmap_size = (parsed_args.env_map_w, parsed_args.env_map_h)

    if (parsed_args.load_frames is False) and (parsed_args.image_size != ''):
            parsed_args.image_size = decode_str_tuple(parsed_args.image_size)

    return parsed_args


class ConfigContext(object):
    """
    Class to manage the active current configuration, creates temporary `yaml`
    file containing the configuration currently being used so it can be
    accessed anywhere.
    """
    # parsed_args = parse_args(sys.argv[1:])

    def __init__(self):
        _validstr = string.ascii_lowercase + string.digits
        time_stamp = time.strftime('%Y.%m.%d_%H%M%S', time.localtime(time.time()))
        random_hash = ''.join([_validstr[i] for i in np.random.randint(0, len(_validstr), 8)])
        active_configs_path = Path(os.path.join(project_dir, "active_configs"))
        active_configs_path.mkdir(parents=True, exist_ok=True)
        self.yaml_unique_id = f"{time_stamp}_{random_hash}"
        self.active_yaml_filename = os.path.join(active_configs_path, f"config_{self.yaml_unique_id}.yaml")

        self.parsed_args = parse_args(sys.argv[1:])

    def __enter__(self):
        self.clean()
        # store all the parsed_args in a yaml file
        # with open(self.active_yaml_filename, 'w') as f:
        #     yaml.dump(self.parsed_args.__dict__, f)

        # do the same in log_dir, if it is valid
        # if self.parsed_args.log_dir:
        #     log_dir_path = Path(os.path.abspath(self.parsed_args.log_dir))
        #     log_dir_path.mkdir(parents=True, exist_ok=True)
        #     log_yaml_filename = os.path.join(log_dir_path, f"config_{self.yaml_unique_id}.yaml")
        #     with open(log_yaml_filename, 'w') as f:
        #         yaml.dump(self.parsed_args.__dict__, f)

        return self.parsed_args

    def clean(self):
        if os.path.exists(self.active_yaml_filename):
            # remove current active yaml file, if exists
            os.remove(self.active_yaml_filename)

    def __exit__(self, exception_type, exception_value, traceback):
        self.clean()
