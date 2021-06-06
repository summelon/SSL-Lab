import copy
import torch
from omegaconf import DictConfig
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate

from .base import BaseModel
from ..models.simsiam_arm import SiameseArm


class SimEstimatorModel(BaseModel):
    def __init__(
        self,
        basic: DictConfig,
        backbone: DictConfig,
        mlp: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.online_network = self._prepare_model()
        self.target_network = self._prepare_model()
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.weight_callback = BYOLMAWeightUpdate()

        self.criterion = FeatureCrossEntropy(
            target_temp=self.hparams.basic.target_temp,
            online_temp=self.hparams.basic.online_temp,
        )
        self.outputs = None

    def forward(self, x):
        # Input: x0, x1, return_features
        # Output: ((projection, prediction), backbone_feature)
        _, backbone_feature = self.online_network(x, return_features=True)
        return backbone_feature

    def _asymmetric_loss(self, a, b):
        online_z, online_p = self.online_network(a)
        with torch.no_grad():
            target_z, target_p = self.target_network(b)
        self.outputs = online_p
        # Symmetric loss calculation here
        return self.criterion(online_p, target_p)

    def training_step(self, batch, batch_idx):
        # (x0, x1), _, _ = batch
        # (Aug0, Aug1, w/o aug), label
        (x0, x1, _), _ = batch

        loss_x0 = self._asymmetric_loss(x0, x1)
        loss_x1 = self._asymmetric_loss(x1, x0)
        loss_tot = 0.5 * (loss_x0 + loss_x1)
        self.log_dict(
            {'loss_x0': loss_x0, 'loss_x1': loss_x1},
            prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True,
        )
        return loss_tot

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader):
        self.weight_callback.on_train_batch_end(
            self.trainer, self, outputs, batch, batch_idx, dataloader)

    def _prepare_model(self):
        online_network = SiameseArm(
            backbone=self.hparams.backbone.backbone,
            first_conv=self.hparams.backbone.first_conv,
            maxpool1=self.hparams.backbone.maxpool1,
            proj_hidden_dim=self.hparams.mlp.proj_hidden_dim,
            pred_hidden_dim=self.hparams.mlp.pred_hidden_dim,
            out_dim=self.hparams.mlp.out_dim,
            num_proj_mlp_layer=self.hparams.mlp.num_proj_mlp_layer,
            proj_last_bn=self.hparams.mlp.proj_last_bn,
            using_predictor=self.hparams.mlp.using_predictor,
            linear_as_pred=self.hparams.mlp.linear_as_pred,
            norm=self.hparams.mlp.norm,
            num_groups=self.hparams.mlp.num_groups,
            pred_last_norm=self.hparams.mlp.pred_last_norm,
        )
        return online_network


class FeatureCrossEntropy(torch.nn.Module):
    def __init__(self, online_temp, target_temp):
        super().__init__()
        self.target_temp = target_temp
        self.online_temp = online_temp

    def forward(self, online_pred, target_pred):
        target_pred = target_pred.detach()
        target_pred = torch.softmax(target_pred/self.target_temp, dim=0)
        online_pred = torch.softmax(online_pred/self.online_temp, dim=0)
        cross_entropy = -(online_pred * torch.log(target_pred)).sum(dim=0)
        return cross_entropy.mean()
