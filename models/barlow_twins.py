import math
import torch
import lightly
import torchvision
import pytorch_lightning as pl
from pl_bolts.optimizers.lars_scheduling import LARSWrapper


class BarlowTwins(pl.LightningModule):
    def __init__(
            self,
            base_lr: float,
            weight_decay: float,
            momentum: float,
            eff_batch_size: int,
            warm_up_steps: int,
            max_steps: int,
            num_classes: int,
            # Model specific args:
            mlp_config: dict,
            arch: str = 'resnet18',
    ):
        super().__init__()
        self.hparams = dict(
            base_lr=base_lr,
            weight_decay=weight_decay,
            momentum=momentum,
            arch=arch,
            warm_up_steps=warm_up_steps,
            max_steps=max_steps,
            eff_batch_size=eff_batch_size,
            num_classes=num_classes,
            mlp_config=mlp_config,
        )
        self.online_network = self._prepare_model()
        self.criterion = lightly.loss.BarlowTwinsLoss()
        self.outputs = None

    def forward(self, x):
        # Input: x0, x1, return_features
        # Output: ((projection, prediction), backbone_feature)
        _, backbone_feature = self.online_network(x, return_features=True)
        return backbone_feature

    def training_step(self, batch, batch_idx):
        # (x0, x1), _, _ = batch
        # (Aug0, Aug1, w/o aug), label
        (x0, x1, _), _ = batch
        y0, y1 = self.online_network(x0, x1)
        loss = self.criterion(y0, y1)
        # TODO: modify to more clean method
        self.outputs = y0
        self.log(
            'repr_loss', loss, prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        # scaled_lr = self.hparams.base_lr * self.hparams.eff_batch_size / 256
        # optimizer = torch.optim.Adam(
        optimizer = torch.optim.SGD(
            self.online_network.parameters(),
            momentum=self.hparams.momentum,
            lr=self.hparams.base_lr,
            weight_decay=self.hparams.weight_decay
        )
        optimizer = LARSWrapper(optimizer)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self._custom_scheduler_fn()),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def _custom_scheduler_fn(self):
        max_steps = self.hparams.max_steps

        def _cosine_decay_scheduler(global_step):
            if global_step < self.hparams.warm_up_steps:
                lr_factor = global_step / self.hparams.warm_up_steps
            else:
                global_step = min(global_step, max_steps)
                lr_factor = \
                    0.5 * (1 + math.cos(math.pi * global_step / max_steps))
            return lr_factor
        return _cosine_decay_scheduler

    def _prepare_model(self):
        from pl_bolts.utils.semi_supervised import Identity
        resnet = torchvision.models.__dict__[self.hparams.arch]()
        num_features = resnet.fc.in_features
        resnet.fc = Identity()
        online_network = lightly.models.BarlowTwins(
            resnet,
            num_ftrs=num_features,
            proj_hidden_dim=self.hparams.mlp_config['proj_hidden_dim'],
            out_dim=self.hparams.mlp_config['out_dim'],
            # Only the layers number of projection module is different
            num_mlp_layers=self.hparams.mlp_config['num_proj_layers'],
        )
        return online_network
