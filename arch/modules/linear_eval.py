import torch
import pytorch_lightning as pl
from pl_bolts.models.self_supervised import resnets
from .base import BaseModel


class LinearEvalModel(BaseModel):
    def __init__(
        self,
        base_lr: float,
        weight_decay: float,
        momentum: float,
        eff_batch_size: int,
        warm_up_steps: int,
        max_steps: int,
        num_classes: int,
        ckpt_path: str,
        backbone: str,
        optimizer: str = "adam",
    ):
        super().__init__()
        self.save_hyperparameters()
        state_dict = torch.load(ckpt_path)
        self.maxpool1 = state_dict["hyper_parameters"]["maxpool1"]
        self.first_conv = state_dict["hyper_parameters"]["first_conv"]

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
        self.backbone = getattr(resnets, self.hparams.backbone)(
            maxpool1=self.maxpool1,
            first_conv=self.first_conv,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.fc = torch.nn.Linear(
            self.backbone.fc.in_features,
            self.hparams.num_classes,
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
