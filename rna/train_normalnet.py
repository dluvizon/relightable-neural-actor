
import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np

from .cnnmodels import NormalNet
from .dataloader import NeuraNormVisDataloader
from .utils import Tensorboard_NVidiaSMI
from .losses import NeuraNormalLoss

import lightning as L


class LitNormalNet(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

        # build the loss function
        self.loss_fn = NeuraNormalLoss()

        # Cosmetics...
        self.checkpoints_path = os.path.join(args.log_dir, 'checkpoints', args.experiment)
        Path(self.checkpoints_path).mkdir(parents=True, exist_ok=True)
        self.textures_path = os.path.join(args.log_dir, 'textures', args.experiment)
        Path(self.textures_path).mkdir(parents=True, exist_ok=True)

        self.init_tensorboard(args)


    def training_step(self, data, batch_idx):
        # Get the data and arrange it for training
        normals_in = data['norm_partial'].permute(0, 3, 1, 2)
        normals_in_mask = data['norm_partial_mask'].unsqueeze(1)
        normals_target = data['norm_full'].permute(0, 3, 1, 2)
        normals_target_mask = data['norm_full_mask'].unsqueeze(1)

        # Forward the model
        normals_out, normals_out_mask = model(normals_in, normals_in_mask)

        # Compute loss
        loss_normals_l2, loss_normals_vgg = self.loss_fn(normals_out, normals_target, normals_target_mask)
        loss = loss_normals_l2 + 0.002 * loss_normals_vgg

        loss_meta = {
            'input_normals': normals_in_mask[0:1] * (normals_in[0:1] / 2 + 0.5),
            'pred_normals': normals_out_mask[0:1] * (normals_out[0:1] / 2 + 0.5),
            'gt_normals': normals_target_mask[0:1] * (normals_target[0:1] / 2 + 0.5),
            'pred_new': (normals_target_mask[0:1] - normals_in_mask[0:1]) * (normals_out[0:1] / 2 + 0.5),
        }
        if (self.global_rank == 0) and (self.local_rank == 0):
            self.update_tensorboard(loss_normals_l2, loss_normals_vgg, loss_meta)

        return loss

    def on_train_epoch_end(self) -> None:
        model_fname = f'normalnet_weights_{self.current_epoch + 1:03d}'
        torch.save(model.state_dict(), os.path.join(self.checkpoints_path, model_fname + '.pt'))

        return super().on_train_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.args.lr)
        return optimizer

    def init_tensorboard(self, args):
        self.tensorboard_path = os.path.join(args.log_dir, 'tensorboard', args.experiment)
        Path(self.tensorboard_path).mkdir(parents=True, exist_ok=True)

        self.tb_writer = SummaryWriter(log_dir=self.tensorboard_path, comment=args.experiment)
        print (f"Logging tensorboard at '{self.tensorboard_path}':'{args.experiment}'")

        self.nvidiasmi_logger = Tensorboard_NVidiaSMI()
        self.nvidiasmi_logger.start()

        self.err_n_l2 = []
        self.err_n_vgg = []

    def update_tensorboard(self, loss_n_l2, loss_n_vgg, loss_meta):
        self.err_n_l2.append(float(loss_n_l2.detach().cpu().numpy()))
        self.err_n_vgg.append(float(loss_n_vgg.detach().cpu().numpy()))

        if self.global_step % 100 == 0:
            step = self.global_step
            self.tb_writer.add_scalar("Loss/train_normals_L2", np.mean(self.err_n_l2), step)
            self.tb_writer.add_scalar("Loss/train_normals_vgg", np.mean(self.err_n_vgg), step)
            self.err_n_l2 = []
            self.err_n_vgg = []

            util_gpu, used_mem, memory_io = self.nvidiasmi_logger.get_info()
            for gpu_idx, (gpu, used, io) in enumerate(zip(util_gpu, used_mem, memory_io)):
                self.tb_writer.add_scalar(f"gpu/cuda:{gpu_idx}", gpu, step)
                self.tb_writer.add_scalar(f"mem/cuda:{gpu_idx}", used / (1024 * 1024 * 1024), step) # show in GB
                self.tb_writer.add_scalar(f"io/cuda:{gpu_idx}", io, step)

            for k in loss_meta.keys():
                im_grid = torchvision.utils.make_grid(loss_meta[k])
                self.tb_writer.add_image(f"Images/{k}", im_grid, step)


if __name__ == '__main__':
    from .config import ConfigContext

    with ConfigContext() as args:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        dataset = NeuraNormVisDataloader(args.data_root, args.train_frames,
                                         training=True,
                                         num_iter_per_epoch=args.num_iter_per_epoch,
                                         load_visibility=False)

        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=0)

        model = NormalNet()
        if args.normalnet_weights != '':
            model.load_state_dict(torch.load(args.normalnet_weights), strict=True)
        model.summary()

        trainer = L.Trainer(max_steps=len(dataset) * args.epochs / args.batch_size,
                            enable_checkpointing=False,
                            strategy='ddp_find_unused_parameters_true',
                            accelerator="gpu", devices="auto")
        trainer.fit(LitNormalNet(model, args), dataloader)
