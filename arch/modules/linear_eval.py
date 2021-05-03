import torch
import pytorch_lightning as pl
from pl_bolts.models.self_supervised import resnets
from omegaconf import DictConfig

from .base import BaseModel


class LinearEvalModel(BaseModel):
    def __init__(
        self,
        basic: str,
        backbone: DictConfig,
        mlp: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        state_dict = torch.load(self.hparams.basic.ckpt_path)

        self._prepare_model()
        self._load_state_dict_to_specific_part(self.backbone, state_dict)
        self.criterion = torch.nn.CrossEntropyLoss()
        # TODO metrics.Accuracy may be wrong over v1.2
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        out = self.backbone(x)[0]
        out = self.backbone.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        return self._share_step("train", batch, self.train_acc)

    def validation_step(self, batch, batch_idx):
        return self._share_step("val", batch, self.val_acc)

    def test_step(self, batch, batch_idx):
        return self._share_step("test", batch, self.test_acc)

    def _share_step(self, stage: str, batch, acc_metric):
        # (x0, x1), _, _ = batch
        # (Aug0, Aug1, w/o aug), label
        (_, _, imgs), lbls = batch
        preds = self(imgs)
        loss = self.criterion(preds, lbls)
        self.log(
            f'test_{stage}_acc', acc_metric(preds, lbls),
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def _prepare_model(self):
        self.backbone = getattr(resnets, self.hparams.backbone.backbone)(
            maxpool1=self.hparams.backbone.maxpool1,
            first_conv=self.hparams.backbone.first_conv,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.fc = torch.nn.Linear(
            self.backbone.fc.in_features,
            self.hparams.basic.num_classes,
        )
        return
