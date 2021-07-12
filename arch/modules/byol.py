import torch
from copy import deepcopy
from omegaconf import DictConfig
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate

from .simsiam import SimSiamModel
from loss.multi_crop_wrapper import MultiCropLossWrapper


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
        # Initial tau as 0.9995 under bs=512 setting according to paper
        # self.weight_callback = BYOLMAWeightUpdate(initial_tau=0.9995)
        self.weight_callback = BYOLMAWeightUpdate()

        self.criterion = MultiCropLossWrapper(
            num_crops=basic.num_local_crops+basic.num_global_crops,
            loss_obj=NormedMSELoss(),
        )
        return

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # Except the first one(weak augmentation)
        images = images[1:]

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
                # Asymmetric forward in Target Network
                use_projector_feature=True,
            )
        loss_tot = self.criterion(
            student_preds=online_features,
            teacher_preds=target_features,
        )

        self.log_dict(
            {"repr_loss": loss_tot},
            prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True,
        )
        return loss_tot

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader):
        self.weight_callback.on_train_batch_end(
            self.trainer, self, outputs, batch, batch_idx, dataloader)


class NormedMSELoss(torch.nn.CosineSimilarity):
    def __init__(self):
        super().__init__()
        # TODO: check the loss implements in lightly and jax
        # Ref: https://github.com/deepmind/deepmind-research/blob/d8df4155dc72fc3dfff9f857412168993348ca23/byol/utils/helpers.py#L109
        # Ref: https://github.com/lightly-ai/lightly/blob/master/lightly/loss/sym_neg_cos_sim_loss.py
        return

    def forward(self, a, b):
        # |x0 - x1| = 2 - 2cos(x0, x1)
        return -2 * super().forward(a, b).mean()
