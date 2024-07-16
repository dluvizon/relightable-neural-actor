
import os

import numpy as np
import torch
from tqdm import tqdm

from .model import NeuRA
from .dataloader import NeuraValidationDataloader
from .utils import post_process_uv_geometry
from .io import save_uv_geometry
from .io import save_prediction_data


if __name__ == '__main__':
    from .config import ConfigContext

    with ConfigContext() as args:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset = NeuraValidationDataloader(args)
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=1, shuffle=False, num_workers=1)

        first_cam  = dataset.cameras[0]
        mid_cam = dataset.cameras[7]
        last_cam = dataset.cameras[-1]
        output_path = os.path.join(args.log_dir, args.experiment)

        model = NeuRA(args, dataset.faces, dataset.face_uv,
                skinning_weights=dataset.skinning_weights,
                tpose_RT=dataset.tpose_RT, uv_mask=dataset.uv_mask,
                canonical_mesh_bounds=dataset.canonical_mesh_bounds)

        if args.geometry_weights != '':
            model.load_state_dict(torch.load(args.geometry_weights), strict=True)

        model.to(device)
        model.eval()
        model.summary()

        def prep(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return x

        with torch.no_grad():
            for f, data in enumerate(tqdm(dataloader)):
                # transfer data to device
                for key in data.keys():
                    data[key] = prep(data[key])

                data['size'] = (data['rgb'].shape[2:0:-1]) # (W, H)
                frame = int(data['sample'][0].numpy())
                cam = int(data['sample'][1].numpy())

                if cam == first_cam:
                    # Init UV buffers
                    tex_h, tex_w = data['texture'].shape[1:3]
                    uv_geometry = {
                        'uv_normals': prep(torch.zeros((1, tex_h * tex_w, 3))),
                        'uv_vis': prep(torch.zeros((1, tex_h * tex_w // 4, np.prod(model.envmap_size)))),
                        'uv_acc': prep(torch.zeros((1, tex_h * tex_w // 4))),
                    }

                # forward the model
                pred = model(**data, compute_normals=True, compute_uvmap_geometry=True, **uv_geometry)
                # save_prediction_data(os.path.join(output_path, f'{frame:06d}_{cam:03d}'), data, pred)

                if cam == mid_cam:
                    uv_normals, uv_normals_mask, uv_vis, uv_vis_mask = \
                        post_process_uv_geometry(**uv_geometry,
                            uv_mask=np.flip(dataset.uv_mask, 0), tex_size=(tex_w, tex_h), median_blur=False)
                    save_uv_geometry(output_path, uv_normals, uv_normals_mask, uv_vis, uv_vis_mask, frame=frame, postfix='mid')

                if cam == last_cam:
                    uv_normals, uv_normals_mask, uv_vis, uv_vis_mask = \
                        post_process_uv_geometry(**uv_geometry,
                            uv_mask=np.flip(dataset.uv_mask, 0), tex_size=(tex_w, tex_h), median_blur=False)
                    save_uv_geometry(output_path, uv_normals, uv_normals_mask, uv_vis, uv_vis_mask, frame=frame, postfix='all')


        print (f"Done.")