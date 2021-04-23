import math
import torch
import lightly

from .base import BaseModel
from ..models.simsiam_arm import SiameseArm


class SimSiamModel(BaseModel):
    def __init__(
        self,
        base_lr: float,
        weight_decay: float,
        momentum: float,
        eff_batch_size: int,
        warm_up_steps: int,
        max_steps: int,
        num_classes: int,
        maxpool1: bool,
        first_conv: bool,
        mlp_config: dict,
        backbone: str = "resnet18",
        optimizer: str = "adam",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.online_network = self._prepare_model()
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
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
        self.outputs = y0[0]
        self.log(
            "repr_loss", loss, prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True,
        )
        return loss

    def _prepare_model(self):
        online_network = SiameseArm(
            backbone=self.hparams.backbone,
            first_conv=self.hparams.first_conv,
            maxpool1=self.hparams.maxpool1,
            proj_hidden_dim=self.hparams.mlp_config["proj_hidden_dim"],
            pred_hidden_dim=self.hparams.mlp_config["pred_hidden_dim"],
            out_dim=self.hparams.mlp_config["out_dim"],
            num_proj_mlp_layer=self.hparams.mlp_config["num_proj_mlp_layer"],
            proj_last_bn=self.hparams.mlp_config["proj_last_bn"],
            using_predictor=self.hparams.mlp_config["using_predictor"],
        )
        return online_network
