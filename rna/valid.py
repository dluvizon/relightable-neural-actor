import os

import torch
from tqdm import tqdm
import copy
from pathlib import Path
import numpy as np
from PIL import Image

from .model import NeuRA
from .dataloader import NeuraValidationDataloader
from .io import save_prediction_data


def load_weights_from_neuralactor(model, weights_filepath):
    state_dict = torch.load(weights_filepath)
    new_state_dict = copy.deepcopy(state_dict['model'])
    for key in state_dict['model'].keys():
        if 'encoder.' in key[:9]:
            new_state_dict[key.replace('encoder.', '', 1)] = new_state_dict.pop(key)
        if 'field.' in key[:7]:
            new_state_dict[key.replace('field.', '', 1)] = new_state_dict.pop(key)
        if 'joint_filter' in key:
            new_state_dict['pe_local_fn.emb'] = new_state_dict.pop('joint_filter.emb')
        if 'den_filters.pos' in key:
            new_state_dict['pe_canonical_fn.emb'] = new_state_dict.pop('den_filters.pos.emb')
        if 'tex_filters.ray' in key:
            new_state_dict['pe_ray_fn.emb'] = new_state_dict.pop('tex_filters.ray.emb')

    del new_state_dict['face_uv']
    del new_state_dict['weights']
    del new_state_dict['vertices']
    del new_state_dict['faces']
    del new_state_dict['bg_color.bg_color']

    for k in model.state_dict().keys():
        if k not in new_state_dict:
            print (f'WARNING! Key {k} not found in NA pretrained model!')

    model.load_state_dict(new_state_dict, strict=True)


if __name__ == '__main__':
    from .config import ConfigContext

    with ConfigContext() as args:

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        output_path = os.path.join(args.log_dir, args.experiment)
        Path(output_path).mkdir(exist_ok=True, parents=True)

        dataset = NeuraValidationDataloader(args)
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=1, shuffle=False, num_workers=2)

        model = NeuRA(args, dataset.faces, dataset.face_uv,
                skinning_weights=dataset.skinning_weights,
                tpose_RT=dataset.tpose_RT, relight=args.relight,
                uv_mask=dataset.uv_mask, uv_deform_mask=dataset.uv_deform_mask,
                canonical_mesh_bounds=dataset.canonical_mesh_bounds)

        if args.relight:
            if args.relight_weights != '':
                model.load_geometry_weights(args.relight_weights, requires_grad=False)
                model.load_radiance_weights(args.relight_weights, requires_grad=False)
                model.load_relight_weights(args.relight_weights)
            else:
                if args.geometry_weights != '':
                    model.load_geometry_weights(args.geometry_weights, requires_grad=False)
                    model.load_radiance_weights(args.geometry_weights, requires_grad=False)
                if args.normvis_weights != '':
                    model.normvis_cnn.load_state_dict(torch.load(args.normvis_weights), strict=True)

        elif args.geometry_weights != '':
            model.load_state_dict(torch.load(args.geometry_weights), strict=False)

        # If path is given, overwrites normalnet and visibility net
        if args.use_uv_nets and (args.normalnet_weights != ''):
            model.normalnet.load_state_dict(torch.load(args.normalnet_weights), strict=True)
        if args.use_uv_nets and (args.visibilitynet_weights != ''):
            model.visibilitynet.load_state_dict(torch.load(args.visibilitynet_weights), strict=True)

        if (args.overwrite_albedo_map != "") or (args.overwrite_brdf_map != ""):
            albedo_map_path = args.overwrite_albedo_map if args.overwrite_albedo_map != "" else None
            brdf_map_path = args.overwrite_brdf_map if args.overwrite_brdf_map != "" else None
            model.set_albedo_brdf_maps(albedo_map_path, brdf_map_path)

        model.to(device)
        model.eval()
        model.summary()

        # If in static albedo mode, export the albedo and BRDF UV maps
        if args.static_albedo:
            Path(output_path).mkdir(parents=True, exist_ok=True)

            # static_albedo = torch.sigmoid(model.static_albedo_map).flip(2)
            static_albedo = model.static_albedo_map.flip(2)
            static_albedo = static_albedo.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
            fname = os.path.join(output_path, f'static_albedo.png')
            Image.fromarray((255 * static_albedo).astype(np.uint8)).save(fname)

            # static_brdf = torch.sigmoid(model.static_brdf_map).flip(2)
            static_brdf = model.static_brdf_map.flip(2)
            static_brdf = static_brdf.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
            fname = os.path.join(output_path, f'static_brdf.png')
            Image.fromarray((255 * static_brdf.squeeze(-1)).astype(np.uint8)).save(fname)

        def prep(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return x

        # for _ in range(28):
            # model.rotate_envmap_one_step()
            # model.rotate_envmap_high_res(steps=1024//32)

        with torch.no_grad():
            for f, data in enumerate(tqdm(dataloader)):
                # transfer data to device
                for key in data.keys():
                    data[key] = prep(data[key])

                if args.load_frames:
                    data['size'] = (data['rgb'].shape[2:0:-1]) # (W, H)
                    tmp_mask = data['mask']
                else:
                    img_w, img_h = args.image_size
                    data['size'] = (img_w // args.valid_res_subsample, img_h // args.valid_res_subsample)
                    tmp_mask = None

                frame = int(data['sample'][0].numpy())
                cam = int(data['sample'][1].numpy())
                data['mask'] = None

                # forward the model
                pred = model(**data, compute_normals=True)

                # if args.rotate_envmap:
                #     model.rotate_envmap_high_res()
                # if f % 5 == 0:
                # model.rotate_envmap_one_step()

                data['mask'] = tmp_mask
                save_prediction_data(output_path, data, pred,
                                     validation=True, envmap=model.envmap_light)

        print (f"Done.")