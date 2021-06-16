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
        self.weight_callback = BYOLMAWeightUpdate()
        self.prototypes = LinearTransform(
            input_dim=self.hparams.mlp.out_dim,
            output_dim=self.hparams.mlp.k_dim,
            last_norm=self.hparams.mlp.pred_last_norm,
            norm=self.hparams.mlp.norm,
            dino_last=self.hparams.mlp.dino_last,
        )

        # Cross-Entropy term
        self.criterion = FeatureCrossEntropy(
            target_temp=self.hparams.basic.target_temp,
            online_temp=self.hparams.basic.online_temp,
            center_momentum=self.hparams.basic.center_momentum,
        )
        # Regularization term
        prototype_params = self.prototypes.named_parameters()
        prototype_weight_v = dict(prototype_params)["linear_trans.0.weight_v"]
        self.regularization = FeatureIsolation(
            weight_v=prototype_weight_v,
            factor_lambda=self.hparams.basic.factor_lambda,
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

        # Online network
        (_, o0), (_, o1) = self.online_network(x0), self.online_network(x1)
        o0, o1 = self.prototypes(o0), self.prototypes(o1)
        # Target network
        with torch.no_grad():
            (t0, _), (t1, _) = self.target_network(x0), self.target_network(x1)
            t0, t1 = self.prototypes(t0), self.prototypes(t1)
        # Loss
        loss, collapse = self.criterion((o0, o1), (t0, t1))
        independent = self.regularization()
        loss_tot = loss + independent

        self.log_dict(
            {"ce": loss, "regularization": independent, "entropy": collapse},
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

    def _filter_params(self):
        params = (list(self.online_network.parameters())
                  + list(self.prototypes.parameters()))
        # Exclude biases and bn
        if self.hparams.optimizer == "lars":
            params = self._exclude_from_wt_decay(
                params,
                weight_decay=self.hparams.weight_decay
            )
        return params


class FeatureCrossEntropy(torch.nn.Module):
    def __init__(
        self,
        online_temp,
        target_temp,
        center_momentum,
    ):
        super().__init__()
        self.target_temp = target_temp
        self.online_temp = online_temp
        self.center_momentum = center_momentum

    def _asymmetric_loss(self, online_pred, target_pred):
        target_pred = target_pred.detach()
        online_pred = online_pred / self.online_temp
        target_pred = torch.softmax(target_pred/self.target_temp, dim=1)

        cross_entropy = \
            -(target_pred * torch.log_softmax(online_pred, dim=1)).sum(dim=1)
        with torch.no_grad():
            info_entropy = -(target_pred * torch.log(target_pred)).sum(dim=1)

        return cross_entropy.mean(), info_entropy.mean()

    def forward(self, online_preds, target_preds):
        ce1, ie1 = self._asymmetric_loss(online_preds[0], target_preds[1])
        ce2, ie2 = self._asymmetric_loss(online_preds[1], target_preds[0])

        loss = 0.5 * (ce1 + ce2)
        collapse = 0.5 * (ie1 + ie2)
        return loss, collapse


class FeatureIsolation(torch.nn.Module):
    def __init__(
        self,
        weight_v,
        factor_lambda,
    ):
        super().__init__()
        self.factor_lambda = factor_lambda
        self.weight_v = weight_v
        self.K, self.D = weight_v.shape
        self.diag_mask = ~torch.eye(self.K, dtype=bool, device=weight_v.device)
        return

    def forward(self):
        # Unit norm
        normed_w = self.weight_v / self.weight_v.norm(dim=1, keepdim=True)
        # Normalization
        # mean_w = self.weight_v.mean(dim=1, keepdim=True)
        # std_w = self.weight_v.std(dim=1, keepdim=True)
        # normed_w = (self.weight_v - mean_w) / std_w

        # Divide D in BarlowTwins paper
        # similarity = torch.mm(normed_w, normed_w.T) / self.D
        similarity = torch.mm(normed_w, normed_w.T)

        # Use abs for stable loss
        loss = similarity[self.diag_mask].abs()
        # loss = similarity[self.diag_mask].pow(2)

        # Divide 2 since triangle symmetric
        loss_tot = loss.mean() * self.factor_lambda
        # loss_tot = loss.sum() * self.factor_lambda
        return loss_tot
