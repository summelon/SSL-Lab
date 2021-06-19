import copy
import torch
from omegaconf import DictConfig
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate

from .base import BaseModel
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
        self.criterion = FeatureCrossEntropy(k_dim=self.hparams.mlp.k_dim)
        # Regularization term
        self.regularization = FeatureIsolation(prototype_layer=self.prototypes)
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

        # Online network(using predictor output)
        (_, o0), (_, o1) = self.online_network(x0), self.online_network(x1)
        o0, o1 = self.prototypes(o0), self.prototypes(o1)
        # Target network(using projector output)
        with torch.no_grad():
            (t0, _), (t1, _) = self.target_network(x0), self.target_network(x1)
            t0, t1 = self.prototypes(t0), self.prototypes(t1)
        # Loss
        loss, anneal, t_entropy, o_entropy = self.criterion((o0, o1), (t0, t1))
        independence = self.regularization()
        loss_tot = loss + independence + anneal

        self.log_dict(
            {"ce": loss,
             "target_entropy": t_entropy,
             "online_entropy": o_entropy,
             "annealing": anneal,
             "scale_s": self.criterion.scale_s,
             "regularization": independence},
            prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True,
        )
        self.outputs = o0
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


class FeatureCrossEntropy(torch.nn.Module):
    def __init__(
        self,
        k_dim,
    ):
        super().__init__()
        from math import ceil, log10
        self.scale_s = torch.nn.Parameter(torch.ones(1))
        self.lower_bound = 4 + 2 * ceil(log10(k_dim))
        return

    def _asymmetric_loss(self, online_pred, target_pred):
        target_pred = target_pred.detach()
        target_pred = torch.softmax(target_pred*self.scale_s, dim=1)
        online_pred = online_pred * self.scale_s
        online_log_softmax = torch.log_softmax(online_pred, dim=1)

        cross_entropy = \
            -(target_pred * online_log_softmax).sum(dim=1)
        with torch.no_grad():
            t_entropy = -(target_pred * torch.log(target_pred)).sum(dim=1)
            online_pred = torch.softmax(online_pred, dim=1)
            o_entropy = -(online_pred * online_log_softmax).sum(dim=1)

        return cross_entropy.mean(), t_entropy.mean(), o_entropy.mean()

    def forward(self, online_preds, target_preds):
        ce1, te1, oe1 = self._asymmetric_loss(online_preds[0], target_preds[1])
        ce2, te2, oe2 = self._asymmetric_loss(online_preds[1], target_preds[0])

        loss = 0.5 * (ce1 + ce2)
        annealing = 0.5 * (self.lower_bound - self.scale_s).pow(2)
        target_entropy = 0.5 * (te1 + te2)
        online_entropy = 0.5 * (oe1 + oe2)

        return loss, annealing, target_entropy, online_entropy


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
        # Unit norm
        normed_w = self.weight_v / self.weight_v.norm(dim=1, keepdim=True)
        similarity = torch.mm(normed_w, normed_w.T)
        # Symmetric matrix, use the upper triangle only
        loss = similarity.triu(diagonal=1).pow(2)
        return loss.mean()
