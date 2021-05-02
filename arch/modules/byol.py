import torch
from copy import deepcopy
from omegaconf import DictConfig
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate

from .simsiam import SimSiamModel


class BYOLModel(SimSiamModel):
    def __init__(
        self,
        basic: DictConfig,
        backbone: DictConfig,
        mlp: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
    ):
        self.save_hyperparameters()
        super().__init__(**self.hparams)

        self.target_network = deepcopy(self.online_network)
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.criterion = torch.nn.CosineSimilarity()
        # TODO Check how momentum encoder update
        # Initial tau as 0.9995 under bs=512 setting according to paper
        # self.weight_callback = BYOLMAWeightUpdate(initial_tau=0.9995)
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
            {'loss_x0': loss_x0, 'loss_x1': loss_x1},
            prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True,
        )
        return loss_tot

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader):
        self.weight_callback.on_train_batch_end(
            self.trainer, self, outputs, batch, batch_idx, dataloader)
