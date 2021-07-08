import copy
import torch
from omegaconf import DictConfig
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate

from .base import BaseModel
from loss.soft_xent_loss import MultiCropSoftXentLoss
from ..models.simsiam_arm import SiameseArm, LinearTransform


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
        self.target_network = copy.deepcopy(self.online_network)
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.prototypes = LinearTransform(
            input_dim=self.hparams.mlp.out_dim,
            output_dim=self.hparams.mlp.k_dim,
            last_norm=self.hparams.mlp.pred_last_norm,
            norm=self.hparams.mlp.norm,
            dino_last=self.hparams.mlp.dino_last,
        )
        self.weight_callback = BYOLMAWeightUpdate()

        # Cross-Entropy term
        self.criterion = MultiCropSoftXentLoss(
            num_crops=basic.num_local_crops+basic.num_global_crops,
            # Temperature will be assigned by scaling factor during training
            student_temp=1.0,
            teacher_temp=1.0,
            k_dim=self.hparams.mlp.k_dim,
            anneal_temp=True,
        )
        # Regularization term
        self.regularization = FeatureIsolation(prototype_layer=self.prototypes)

    def forward(self, x):
        # Input: x0, x1, return_features
        # Output: ((projection, prediction), backbone_feature)
        _, backbone_feature = self.online_network(x, return_features=True)
        return backbone_feature

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # Except the last one(weak augmentation)
        images = images[:-1]

        # Online
        online_features = self._multi_crop_forward(
            images=images,
            network=self.online_network,
            local_forward=True,
            use_projector_feature=False,
        )
        online_features = self.prototypes(online_features)
        # Target
        with torch.no_grad():
            target_features = self._multi_crop_forward(
                images=images,
                network=self.target_network,
                local_forward=False,
                # Asymmetric forward like BYOL
                use_projector_feature=True,
            )
            target_features = self.prototypes(target_features)
        # Cross-Entropy + annealing loss
        loss = self.criterion(
            student_preds=online_features,
            teacher_preds=target_features,
        )
        independence = self.regularization()
        loss_tot = loss + independence
        self.log_dict(
            {"ce": loss,
             "scale_s": self.criterion.scale_s,
             "regularization": independence},
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
            k_dim=self.hparams.mlp.k_dim,
        )
        return online_network


class FeatureIsolation(torch.nn.Module):
    def __init__(
        self,
        prototype_layer,
    ):
        super().__init__()
        prototype_params = prototype_layer.named_parameters()
        self.weight_v = dict(prototype_params)["linear_trans.0.weight_v"]
        return

    def forward(self):
        # TODO: add centering
        # Unit norm
        normed_w = self.weight_v / self.weight_v.norm(dim=1, keepdim=True)
        similarity = torch.mm(normed_w, normed_w.T)
        # Symmetric matrix, use the upper triangle only
        loss = similarity.triu(diagonal=1).pow(2)
        return loss.mean()
