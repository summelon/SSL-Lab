import torch
from copy import deepcopy
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from pl_bolts.optimizers.lars_scheduling import LARSWrapper

from .simsiam import SimSiamModel


class BYOLModel(SimSiamModel):
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
        super().__init__(
            base_lr=base_lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eff_batch_size=eff_batch_size,
            warm_up_steps=warm_up_steps,
            max_steps=max_steps,
            num_classes=num_classes,
            mlp_config=mlp_config,
            arch=arch,
        )

        self.target_network = deepcopy(self.online_network)
        # Avoid not producing loss problem
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.criterion = torch.nn.CosineSimilarity()
        self.weight_callback = BYOLMAWeightUpdate()

    def _asymmetric_loss(self, a, b):
        online_z, online_p = self.online_network(a)
        with torch.no_grad():
            target_z, target_p = self.target_network(b)
        self.outputs = online_z
        return -2 * self.criterion(online_p, target_z).mean()

    def training_step(self, batch, batch_idx):
        # (x0, x1), _, _ = batch
        # (Aug0, Aug1, w/o aug), label
        (x0, x1, _), _ = batch

        loss_x0 = self._asymmetric_loss(x0, x1)
        loss_x1 = self._asymmetric_loss(x1, x0)
        loss_tot = loss_x0 + loss_x1
        self.log_dict(
            {'loss_x0': loss_x0,
             'loss_x1': loss_x1},
            prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True
        )

        return loss_tot

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader):
        self.weight_callback.on_train_batch_end(
            self.trainer, self, outputs, batch, batch_idx, dataloader)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
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
