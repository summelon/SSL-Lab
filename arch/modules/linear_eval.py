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
        self.load_state_dict(state_dict)
        self.criterion = torch.nn.CrossEntropyLoss()
        # TODO metrics.Accuracy may be wrong over v1.2
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        out = self.backbone(x)[0]
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        return self._share_step("train", batch)

    def validation_step(self, batch, batch_idx):
        return self._share_step("val", batch)

    def _share_step(self, stage: str, batch):
        # (x0, x1), _, _ = batch
        # (Aug0, Aug1, w/o aug), label
        (_, _, imgs), lbls = batch
        preds = self(imgs)
        loss = self.criterion(preds, lbls)
        self.log(
            f'test_{stage}_acc', self.accuracy(preds, lbls),
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
        self.fc = torch.nn.Linear(
            self.backbone.fc.in_features,
            self.hparams.basic.num_classes,
        )
        return

    def load_state_dict(self, state_dict):
        dict_zip = zip(
            state_dict["state_dict"].items(),
            self.backbone.state_dict().items()
        )
        match_dict = {}
        for (s_k, s_v), (m_k, m_v) in dict_zip:
            if (m_k in s_k) and (s_v.shape == m_v.shape):
                match_dict[m_k] = s_v
        msg = self.backbone.load_state_dict(match_dict, strict=False)
        print(f"[ INFO ] Missing keys: {msg.missing_keys}")
        return
