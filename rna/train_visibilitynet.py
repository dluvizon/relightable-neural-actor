
import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np

from .cnnmodels import VisibilityNet
from .dataloader import NeuraNormVisDataloader
from .utils import Tensorboard_NVidiaSMI
from .losses import NeuraVisibilityLoss

import lightning as L


class LitNormalNet(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

        # build the loss function
        self.loss_fn = NeuraVisibilityLoss()

        # Cosmetics...
        self.checkpoints_path = os.path.join(args.log_dir, 'checkpoints', args.experiment)
        Path(self.checkpoints_path).mkdir(parents=True, exist_ok=True)
        self.textures_path = os.path.join(args.log_dir, 'textures', args.experiment)
        Path(self.textures_path).mkdir(parents=True, exist_ok=True)

        self.init_tensorboard(args)


    def training_step(self, data, batch_idx):
        # Get the data and arrange it for training
        normals = data['norm_partial'].permute(0, 3, 1, 2)
        normals_mask = data['norm_partial_mask'].unsqueeze(1)
        visibility_in = data['vis_partial'].permute(0, 3, 1, 2)
        visibility_in_mask = data['vis_partial_mask'].unsqueeze(1)
        visibility_target = data['vis_full'].permute(0, 3, 1, 2)
        visibility_target_mask = data['vis_full_mask'].unsqueeze(1)

        # Forward the model
        visibility_out, visibility_mask_out = model(visibility_in, visibility_in_mask, normals, normals_mask)

        # # Compute the loss
        loss = self.loss_fn(visibility_out, visibility_target, visibility_target_mask)

        loss_meta = {
            'input_visibility': visibility_in_mask[0:1] * visibility_in[0:1].mean(dim=1, keepdims=True),
            'pred_visibility': visibility_mask_out[0:1] * visibility_out[0:1].mean(dim=1, keepdims=True),
            'gt_visibility': visibility_target_mask[0:1] * visibility_target[0:1].mean(dim=1, keepdims=True),
            'pred_new': (visibility_target_mask[0:1] - visibility_in_mask[0:1]) * visibility_out[0:1].mean(dim=1, keepdims=True),
        }
        self.update_tensorboard(loss, loss_meta)

        return loss

    def on_train_epoch_end(self) -> None:
        model_fname = f'visibilitynet_weights_{self.current_epoch + 1:03d}'
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

        self.err_bc = []
        self.err_n_grad = []
        self.err_v_l1 = []
        self.err_v_grad = []


    def update_tensorboard(self, loss_bc, loss_meta):
        self.err_bc.append(float(loss_bc.detach().cpu().numpy()))

        if self.global_step % 100 == 0:
            step = self.global_step
            self.tb_writer.add_scalar("Loss/train_loss", np.mean(self.err_bc), step)
            self.err_bc = []

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
                                         load_visibility=True)

        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=4)

        model = VisibilityNet()
        if args.visibilitynet_weights != '':
            model.load_state_dict(torch.load(args.visibilitynet_weights), strict=True)
        model.summary()

        trainer = L.Trainer(max_steps=len(dataset) * args.epochs / args.batch_size,
                            enable_checkpointing=False,
                            strategy='ddp_find_unused_parameters_true',
                            accelerator="gpu", devices="auto")
        trainer.fit(LitNormalNet(model, args), dataloader)
