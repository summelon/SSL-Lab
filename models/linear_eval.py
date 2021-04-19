import math
import torch
import lightly
import torchvision
import pytorch_lightning as pl


class LinearEvalModel(pl.LightningModule):
    def __init__(
            self,
            base_lr: float,
            weight_decay: float,
            eff_batch_size: int,
            max_steps: int,
            warm_up_steps: int,
            num_classes: int,
            arch: str = 'resnet18'
    ):
        super().__init__()
        self.hparams = dict(
            arch=arch,
            base_lr=base_lr,
            weight_decay=weight_decay,
            warm_up_steps=warm_up_steps,
            max_steps=max_steps,
            eff_batch_size=eff_batch_size,
            num_classes=num_classes,
        )

        self._prepare_model(arch, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        # (x0, x1), _, _ = batch
        # (Aug0, Aug1, w/o aug), label
        (_, _, imgs), lbls = batch
        preds = self(imgs)
        loss = self.criterion(preds, lbls)
        self.log(
            'test_train_acc', self.accuracy(preds, lbls),
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
        optimizer = torch.optim.Adam(
            self.backbone.fc.parameters(),
            lr=scaled_lr,
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self._custom_scheduler_fn()),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def _custom_scheduler_fn(self):
        warm_up_steps = self.hparams.warm_up_steps
        max_steps = self.hparams.max_steps

        def _cosine_decay_scheduler(global_step):
            if global_step < warm_up_steps:
                lr_factor = global_step / warm_up_steps
            else:
                global_step = min(global_step, max_steps)
                lr_factor = \
                    0.5 * (1 + math.cos(math.pi * global_step / max_steps))
            return lr_factor
        return _cosine_decay_scheduler

    def _prepare_model(self, arch, num_classes):
        self.backbone = torchvision.models.__dict__[arch]()
        for param in self.backbone.parameters():
            param.requires_grad = False
        num_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(num_features, num_classes)
        return

    def load_state_dict(self, path):
        state_dict = torch.load(path)['state_dict']
        match_dict = {}
        for (s_k, s_v), (m_k, m_v) in \
                zip(state_dict.items(), self.backbone.state_dict().items()):
            if (m_k in s_k) and (s_v.shape == m_v.shape):
                match_dict[m_k] = s_v
        msg = self.backbone.load_state_dict(match_dict, strict=False)
        print(f"[ INFO ] Missing keys: {msg.missing_keys}")
        return
