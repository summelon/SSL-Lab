import torch
import torch.distributed as dist
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
        self.target_network.load_state_dict(self.online_network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.weight_callback = BYOLMAWeightUpdate()

        self.criterion = FeatureCrossEntropy(
            target_temp=self.hparams.basic.target_temp,
            online_temp=self.hparams.basic.online_temp,
            center_momentum=self.hparams.basic.center_momentum,
            out_dim=self.hparams.mlp.out_dim,
        )
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

        (_, o0), (_, o1) = self.online_network(x0), self.online_network(x1)
        (_, t0), (_, t1) = self.target_network(x0), self.target_network(x1)
        loss_tot = self.criterion((o0, o1), (t0, t1))
        self.outputs = o0

        self.log_dict(
            {'loss_tot': loss_tot},
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
            dino_last=self.hparams.mlp.dino_last,
        )
        return online_network


class FeatureCrossEntropy(torch.nn.Module):
    def __init__(
            self,
            online_temp,
            target_temp,
            center_momentum,
            out_dim
    ):
        super().__init__()
        self.target_temp = target_temp
        self.online_temp = online_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def _asymmetric_loss(self, online_pred, target_pred):
        target_pred = target_pred.detach()
        online_pred = torch.softmax(online_pred/self.online_temp, dim=0)
        target_pred = (target_pred - self.center) / self.target_temp

        cross_entropy = \
            -(online_pred * torch.log_softmax(target_pred, dim=0)).sum(dim=0)
        return cross_entropy.mean()

    def forward(self, online_preds, target_preds):
        loss = 0.5 * (
            self._asymmetric_loss(online_preds[0], target_preds[1])
            + self._asymmetric_loss(online_preds[1], target_preds[0])
        )
        self.update_center(torch.cat(target_preds))
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        # TODO: make sure onely world 0 do the all_reduce
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = \
            batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )
        return
