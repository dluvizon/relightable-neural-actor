
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .model import NeuRA
from .dataloader import NeuraValidationDataloader
from .utils import post_process_uv_geometry
from .io import save_uv_geometry


if __name__ == '__main__':
    from .config import ConfigContext

    with ConfigContext() as args:

        image_size = (4112, 3008)
        args.image_size = image_size
        args.load_frames = False


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset = NeuraValidationDataloader(args)

        # Get the UV mask, dilate it for infill, erode it for valid
        kernel = np.ones((3, 3), np.float32)
        valid_uv_mask_d = cv2.dilate(np.flip(dataset.uv_mask, 0), kernel, iterations=3)
        valid_uv_mask_e = cv2.erode(np.flip(dataset.uv_mask, 0), kernel, iterations=2)
        h, w = valid_uv_mask_e.shape[0:2]
        valid_uv_mask_e2 = cv2.resize(valid_uv_mask_e, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

        valid_uv_mask_e = torch.from_numpy(valid_uv_mask_e).to(device).gt(0.5).float().view(1, -1)
        valid_uv_mask_e2 = torch.from_numpy(valid_uv_mask_e2).to(device).gt(0.5).float().view(1, -1)

        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=1, shuffle=False, num_workers=1)

        first_cam  = dataset.cameras[0]
        mid_cam = dataset.cameras[len(dataset.cameras) // 4 - 1]
        last_cam = dataset.cameras[-1]
        output_path = os.path.join(args.log_dir, args.experiment)

        model = NeuRA(args, dataset.faces, dataset.face_uv,
                skinning_weights=dataset.skinning_weights,
                tpose_RT=dataset.tpose_RT, uv_mask=dataset.uv_mask,
                canonical_mesh_bounds=dataset.canonical_mesh_bounds)

        if args.geometry_weights != '':
            model.load_state_dict(torch.load(args.geometry_weights), strict=False)

        model.to(device)
        model.eval()
        model.summary()

        def prep(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return x

        with torch.no_grad():
            for f, data in enumerate(tqdm(dataloader)):
                frame = int(data['sample'][0].numpy())
                cam = int(data['sample'][1].numpy())

                # check if this frame is already computed, if that is the case, skip
                test_fname = os.path.join(output_path, 'neura_visibility', 'dense', f'{frame:06d}.npz')
                if os.path.exists(test_fname):
                    print (f'Skip frame {frame}, camera {cam}')
                    continue

                # transfer data to device
                for key in data.keys():
                    data[key] = prep(data[key])

                # data['size'] = (data['rgb'].shape[2:0:-1]) # (W, H)
                data['size'] = (image_size[0] // args.valid_res_subsample, image_size[1] // args.valid_res_subsample)

                if cam == first_cam:
                    # Init UV buffers
                    tex_h, tex_w = data['texture'].shape[1:3]
                    uv_geometry = {
                        'uv_normals': prep(torch.zeros((1, tex_h * tex_w, 3))),
                        # 'uv_tex': prep(torch.zeros((1, tex_h * tex_w, 3))),
                        # 'uv_tex_cnt': prep(torch.zeros((1, tex_h * tex_w))),
                        'uv_vis': prep(torch.zeros((1, tex_h * tex_w // 4, np.prod(model.envmap_size)))),
                        'uv_acc': prep(torch.zeros((1, tex_h * tex_w // 4))),
                    }

                # forward the model
                pred = model(**data, **uv_geometry, compute_normals=True,
                             compute_uvmap_geometry=True,
                             compute_visibility=args.compute_visibility)
                            #  rgb_tgt=data['rgb'].reshape(1, -1, 3))

                if cam == mid_cam:
                    uv_normals, uv_normals_mask, uv_tex, uv_tex_mask, uv_vis, uv_vis_mask = \
                        post_process_uv_geometry(**uv_geometry,
                            uv_mask=valid_uv_mask_d, tex_size=(tex_w, tex_h))
                    save_uv_geometry(output_path, uv_normals, uv_normals_mask,
                                     uv_tex, uv_tex_mask, uv_vis, uv_vis_mask,
                                     frame=frame, postfix='mid')

                    # uv_normals, uv_normals_mask, uv_tex, uv_tex_mask, uv_vis, uv_vis_mask = \
                    #     post_process_uv_geometry(**uv_geometry,
                    #         uv_mask=valid_uv_mask_d, tex_size=(tex_w, tex_h),
                    #         infill=True, median_blur=False)
                    # save_uv_geometry(output_path, uv_normals, uv_normals_mask,
                    #                  uv_tex, uv_tex_mask, uv_vis, uv_vis_mask,
                    #                  frame=frame, postfix='mid_dense')

                if cam == last_cam:
                    uv_normals, uv_normals_mask, uv_tex, uv_tex_mask, uv_vis, uv_vis_mask = \
                        post_process_uv_geometry(**uv_geometry,
                            uv_mask=valid_uv_mask_d, tex_size=(tex_w, tex_h))
                    save_uv_geometry(output_path, uv_normals, uv_normals_mask,
                                     uv_tex, uv_tex_mask, uv_vis, uv_vis_mask,
                                     frame=frame, postfix='all')

                    uv_geometry['uv_normals'] = valid_uv_mask_e.unsqueeze(-1) * uv_geometry['uv_normals']
                    # uv_geometry['uv_tex_cnt'] = valid_uv_mask_e * uv_geometry['uv_tex_cnt']
                    uv_geometry['uv_acc'] = valid_uv_mask_e2 * uv_geometry['uv_acc']
                    uv_normals, uv_normals_mask, uv_tex, uv_tex_mask, uv_vis, uv_vis_mask = \
                        post_process_uv_geometry(**uv_geometry,
                            uv_mask=valid_uv_mask_d, tex_size=(tex_w, tex_h),
                            infill=True, median_blur=False)
                    save_uv_geometry(output_path, uv_normals, uv_normals_mask,
                                     uv_tex, uv_tex_mask, uv_vis, uv_vis_mask,
                                     frame=frame, postfix='dense')


        print (f"Done.")