import torch
from omegaconf import DictConfig
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate

from .base import BaseModel
from loss.soft_xent_loss import MultiCropSoftXentLoss
from ..models.simsiam_arm import SiameseArm


class DINOModel(BaseModel):
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

        self.criterion = MultiCropSoftXentLoss(
            num_crops=basic.num_local_crops+basic.num_global_crops,
            k_dim=self.hparams.mlp.k_dim,
            teacher_temp=self.hparams.basic.teacher_temp,
            student_temp=self.hparams.basic.student_temp,
            center_momentum=self.hparams.basic.center_momentum,
        )
        self.outputs = None

    def forward(self, x):
        # Input: x0, x1, return_features
        # Output: ((projection, prediction), backbone_feature)
        _, backbone_feature = self.online_network(x, return_features=True)
        return backbone_feature

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # Except the last one(weak augmentation)
        images = images[:-1]

        online_features = self._multi_crop_forward(
            images=images,
            network=self.online_network,
            local_forward=True,
            use_projector_feature=False,
        )
        with torch.no_grad():
            target_features = self._multi_crop_forward(
                images=images,
                network=self.target_network,
                local_forward=False,
                use_projector_feature=False,
            )
        loss_tot = self.criterion(
            student_preds=online_features,
            teacher_preds=target_features,
        )

        self.log_dict(
            {'loss_tot': loss_tot},
            prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True,
        )
        return loss_tot

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader):
        self.weight_callback.on_train_batch_end(
            self.trainer, self, outputs, batch, batch_idx, dataloader)
        return

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
            k_dim=self.hparams.mlp.k_dim,
        )
        return online_network
