import math
import torch
import pytorch_lightning as pl
from pl_bolts.models.self_supervised import resnets


class LinearEvalModel(pl.LightningModule):
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
        self.hparams.maxpool1 = state_dict["hyper_parameters"]["maxpool1"]
        self.hparams.first_conv = state_dict["hyper_parameters"]["first_conv"]

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
        # (x0, x1), _, _ = batch
        # (Aug0, Aug1, w/o aug), label
        (_, _, imgs), lbls = batch
        preds = self(imgs)
        loss = self.criterion(preds, lbls)
        self.log(
            "test_train_acc", self.accuracy(preds, lbls),
            prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        (_, _, imgs), lbls = batch
        preds = self(imgs)
        loss = self.criterion(preds, lbls)
        self.log(
            'test_val_acc', self.accuracy(preds, lbls),
            prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        scaled_lr = self.hparams.base_lr * self.hparams.eff_batch_size / 256
        opt_args = dict(
            params=self.parameters(),
            lr=scaled_lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(**opt_args)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                momentum=self.hparams.momentum, **opt_args)
        else:
            raise NotImplementedError("[ Error ] Optimizer is not implemented")

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self._custom_scheduler_fn()),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def _custom_scheduler_fn(self):
        max_steps = self.hparams.max_steps

        def _cosine_decay_scheduler(global_step):
            if global_step < self.hparams.warm_up_steps:
                lr_factor = global_step / self.hparams.warm_up_steps
            else:
                global_step = min(global_step, max_steps)
                lr_factor = \
                    0.5 * (1 + math.cos(math.pi * global_step / max_steps))
            return lr_factor
        return _cosine_decay_scheduler

    def _prepare_model(self):
        # TODO resnets.fc may be a bug in some situation
        self.backbone = getattr(resnets, self.hparams.backbone)(
            maxpool1=self.hparams.maxpool1,
            first_conv=self.hparams.first_conv,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Only ResNet18/ResNet50 support
        num_features = 512 if self.hparams.backbone == "resnet18" else 2048
        self.fc = torch.nn.Linear(num_features, self.hparams.num_classes)
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
