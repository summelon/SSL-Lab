import copy
import torch
from math import ceil, log10
from omegaconf import DictConfig
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate

from .base import BaseModel
from ..models.simsiam_arm import SiameseArm, LinearTransform
from loss.soft_xent_loss import SoftXentLoss
from loss.multi_crop_wrapper import MultiCropLossWrapper


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

        # Scaling factor for temperature annealing update
        self.scale_s = torch.nn.Parameter(torch.ones(1))
        self.lower_bound = 4 + 2 * ceil(log10(self.hparams.mlp.k_dim))

        # Cross-Entropy term (student & teacher temp default to 1, use scale_s)
        self.criterion = MultiCropLossWrapper(
            num_crops=basic.num_local_crops+basic.num_global_crops,
            loss_obj=SoftXentLoss(
                student_temp=self.hparams.basic.student_temp,
                teacher_temp=self.hparams.basic.teacher_temp,
                teacher_softmax=True,
            )
        )
        # Regularization term
        self.regularization = FeatureIsolation(prototype_layer=self.prototypes)
        return

    def forward(self, x):
        # Input: x0, x1, return_features
        # Output: ((projection, prediction), backbone_feature)
        _, backbone_feature = self.online_network(x, return_features=True)
        return backbone_feature

    def _get_features(self, images):
        # Online
        online_feat = self._multi_crop_forward(
            images=images,
            network=self.online_network,
            local_forward=True,
            use_projector_feature=False,
        )
        # Target
        with torch.no_grad():
            target_feat = self._multi_crop_forward(
                images=images,
                network=self.target_network,
                local_forward=False,
                # Asymmetric forward like BYOL
                use_projector_feature=True,
            )
        return self.prototypes(online_feat), self.prototypes(target_feat)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # Except the first one(weak augmentation)
        images = images[1:]
        online_features, target_features = self._get_features(images)

        # Loss
        ce_loss = self.criterion(
            student_preds=online_features*self.scale_s,
            teacher_preds=target_features*self.scale_s,
        )
        annealing_loss = 0.5 * (self.lower_bound - self.scale_s).pow(2)
        independence = self.regularization()
        loss_tot = ce_loss + independence + annealing_loss

        self.log_dict(
            {"ce": ce_loss,
             "scale_s": self.scale_s,
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
