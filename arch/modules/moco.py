import copy
import torch
from omegaconf import DictConfig
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate

from .base import BaseModel
from ..models.simsiam_arm import SiameseArm
from loss.nt_xent_loss import NTXentLoss
from loss.memory_bank import batch_shuffle_ddp, batch_unshuffle_ddp


class MOCOModel(BaseModel):
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
        self.target_network = copy.deepcopy(self.online_network)
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.criterion = NTXentLoss(
            temperature=self.hparams.basic.temperature,
            memory_bank_size=self.hparams.basic.bank_size,
        )
        self.weight_callback = BYOLMAWeightUpdate(
            initial_tau=self.hparams.basic.tau)
        self.outputs = None

    def forward(self, x):
        # Input: x0, x1, return_features
        # Output: ((projection, prediction), backbone_feature)
        _, backbone_feature = self.online_network(x, return_features=True)
        return backbone_feature

    def _asymmetric_step(self, q_input: torch.Tensor, k_input: torch.Tensor):
        query = self.online_network(q_input)
        self.outputs = query

        # Key features
        with torch.no_grad():
            data_parallel = self.trainer.use_ddp or self.trainer.use_ddp2

            self.weight_callback.update_weights(
                online_net=self.online_network,
                target_net=self.target_network
            )

            if data_parallel:
                k_input, idx_unshuffle = batch_shuffle_ddp(k_input)
            key = self.target_network(k_input)
            if data_parallel:
                key = batch_unshuffle_ddp(key, idx_unshuffle)

        loss = self.criterion(query, key)
        return loss

    def training_step(self, batch, batch_idx):
        # (x0, x1), _, _ = batch
        # (Aug0, Aug1, w/o aug), label
        (x0, x1, _), _ = batch
        loss = 0.5 * (
            self._asymmetric_step(x0, x1) + self._asymmetric_step(x1, x0))

        self.log(
            "repr_loss", loss, prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True,
        )
        return loss

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
        )
        return online_network
