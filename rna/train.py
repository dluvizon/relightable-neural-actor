import os
from pathlib import Path
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import torchvision
import numpy as np


# from .neuralactor import NeuralActor
from .model import NeuRA
from .dataloader import NeuraTrainDataloader
from .losses import NeuraCriterium
from .utils import Tensorboard_NVidiaSMI
from .utils import func_linear2srgb

import lightning as L


class LitNeura(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

        # build the loss function
        self.loss_fn = NeuraCriterium(args)

        # Cosmetics...
        self.checkpoints_path = os.path.join(args.log_dir, 'checkpoints', args.experiment)
        Path(self.checkpoints_path).mkdir(parents=True, exist_ok=True)
        self.textures_path = os.path.join(args.log_dir, 'textures', args.experiment)
        Path(self.textures_path).mkdir(parents=True, exist_ok=True)

        self.init_tensorboard(args)


    def training_step(self, data, batch_idx):
        # forward the model
        args = self.args
        model = self.model

        pred = model(**data, size=args.crop_size)

        static_albedo_map = model.static_albedo_map if args.relight else None
        static_brdf_map = model.static_brdf_map if args.relight else None
        uv_mask = model.uv_mask.flip(2) if args.relight else None

        loss, meta = self.loss_fn(data, pred,
                static_albedo=static_albedo_map,
                static_brdf=static_brdf_map,
                uv_mask=uv_mask)

        if args.relight:
            envmap = func_linear2srgb(model.envmap_light).permute(1, 0) # (3, env_N)
            meta['image/envmap'] = envmap.reshape(1, 3, args.env_map_h, args.env_map_w)

        if self.global_rank == 0:
            self.update_tensorboard(meta)

        return loss

    def on_train_epoch_start(self) -> None:
        # Adjust loss coefficients
        # if self.current_epoch > 0:
        #     self.args.coef_loss_residual *= 0.95
        # print (f'on_train_epoch_start: coef_loss_residual={self.args.coef_loss_residual}')

        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        if (self.global_rank == 0) and (self.local_rank == 0):
            num_weights_to_keep = 3
            save_epoch = self.current_epoch + 1
            model_fname = f'neura_weights_{save_epoch:03d}'
            if self.args.relight:
                model_fname += '_relight'
            torch.save(self.model.state_dict(), os.path.join(self.checkpoints_path, model_fname + '.pt'))
            if save_epoch > num_weights_to_keep:
                delete_weights = f'neura_weights_{save_epoch - num_weights_to_keep:03d}'
                if self.args.relight:
                    delete_weights += '_relight'
                try:
                    os.remove(os.path.join(self.checkpoints_path, delete_weights + '.pt'))
                except Exception as e:
                    print (e)

        return super().on_train_epoch_end()

    def configure_optimizers(self):
        update_frequency = 100
        warming_up_steps = self.args.warming_up_steps
        warming_up_factor = self.args.warming_up_factor

        def scheduler_fn(step):
            s = step * update_frequency
            if s < warming_up_steps:
                factor = warming_up_factor + (1 - warming_up_factor) * (s / warming_up_steps)
            else:
                factor = 1

            return factor

        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = LambdaLR(optimizer, scheduler_fn, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # step or epoch
                "frequency": update_frequency,
            },
        }

    def init_tensorboard(self, args):
        self.tensorboard_path = os.path.join(args.log_dir, 'tensorboard', args.experiment)
        Path(self.tensorboard_path).mkdir(parents=True, exist_ok=True)

        self.tb_writer = SummaryWriter(log_dir=self.tensorboard_path, comment=args.experiment)
        print (f"Logging tensorboard at '{self.tensorboard_path}':'{args.experiment}'")

        self.nvidiasmi_logger = Tensorboard_NVidiaSMI()
        self.nvidiasmi_logger.start()
        self.err_buff_dict = {}

    def update_tensorboard(self, meta):
        for key in meta.keys():
            group = key.split('/')[0]
            if group == 'loss':
                if key not in self.err_buff_dict.keys():
                    self.err_buff_dict[key] = []
                self.err_buff_dict[key].append(meta[key])

        if self.global_step % 100 == 0:
            step = self.global_step
            for key in meta.keys():
                group = key.split('/')[0]
                if group == 'loss':
                    # self.tb_writer.add_scalar(key, meta[key], step)
                    self.tb_writer.add_scalar(key, torch.stack(self.err_buff_dict[key]).mean(), step)
                    self.err_buff_dict[key] = []

                elif group == 'image':
                    im_grid = torchvision.utils.make_grid(meta[key])
                    self.tb_writer.add_image(key, im_grid, step)

            util_gpu, used_mem, memory_io = self.nvidiasmi_logger.get_info()
            for gpu_idx, (gpu, used, io) in enumerate(zip(util_gpu, used_mem, memory_io)):
                self.tb_writer.add_scalar(f"gpu/cuda:{gpu_idx}", gpu, step)
                self.tb_writer.add_scalar(f"mem/cuda:{gpu_idx}", used / (1024 * 1024 * 1024), step) # show in GB
                self.tb_writer.add_scalar(f"io/cuda:{gpu_idx}", io, step)


    def stop(self):
        try:
            self.nvidiasmi_logger.stop()
        except:
            return


if __name__ == '__main__':
    from .config import ConfigContext

    with ConfigContext() as args:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        dataset = NeuraTrainDataloader(args)
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=8)

        model = NeuRA(args, dataset.faces, dataset.face_uv,
                skinning_weights=dataset.skinning_weights,
                tpose_RT=dataset.tpose_RT, relight=args.relight,
                uv_mask=dataset.uv_mask, uv_deform_mask=dataset.uv_deform_mask,
                canonical_mesh_bounds=dataset.canonical_mesh_bounds)
        model.summary()

        if args.relight:
            if args.relight_weights != '':
                model.load_geometry_weights(args.relight_weights, requires_grad=args.train_full_model)
                model.load_radiance_weights(args.relight_weights, requires_grad=args.train_full_model)
                model.load_relight_weights(args.relight_weights)
            elif args.geometry_weights != '':
                model.load_geometry_weights(args.geometry_weights, requires_grad=args.train_full_model)
                model.load_radiance_weights(args.geometry_weights, requires_grad=args.train_full_model)

        elif args.geometry_weights != '':
            model.load_state_dict(torch.load(args.geometry_weights), strict=False)

        # If path is given, overwrites normalnet and visibility net
        if args.use_uv_nets and (args.normalnet_weights != ''):
            model.normalnet.load_state_dict(torch.load(args.normalnet_weights), strict=True)
        if args.use_uv_nets and (args.visibilitynet_weights != ''):
            model.visibilitynet.load_state_dict(torch.load(args.visibilitynet_weights), strict=True)

        lit_model = LitNeura(model, args)
        try:
            trainer = L.Trainer(max_steps=args.epochs * args.num_iter_per_epoch,
                                enable_checkpointing=False,
                                strategy='ddp_find_unused_parameters_true',
                                accelerator="gpu", devices="auto")
            trainer.fit(lit_model, dataloader)

        except Exception as e:
            print ('Exception: ', e)
            print (f'Killing program.')
            sys.stderr.flush()
            sys.stdout.flush()
            os._exit(1)
            # raise
        
        finally:
            lit_model.stop()
            del dataloader
            del trainer
