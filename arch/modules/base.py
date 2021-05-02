import math
import torch
import hydra
import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError(
            "[ Error ] `forward` method not implemented!")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError(
            "[ Error ] `training_step` method not implemented!")

    def _prepare_model(self):
        raise NotImplementedError(
            "[ Error ] `_prepare_model` method not implemented!")

    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            params=self._filter_params(),
            _convert_="all",
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self._custom_scheduler_fn()),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def _custom_scheduler_fn(self):
        max_steps = self.hparams.scheduler.max_steps
        warmup_steps = self.hparams.scheduler.warmup_steps
        base_lr = self.hparams.optimizer.lr
        start_lr = self.hparams.scheduler.start_lr
        eta_min = self.hparams.scheduler.end_lr

        def _cosine_decay_scheduler(global_step):
            if global_step < warmup_steps:
                scaled_linear_range = \
                    (base_lr - start_lr) * (global_step / warmup_steps)
                lr_factor = (start_lr + scaled_linear_range) / base_lr
            else:
                global_step = min(global_step, max_steps)
                progress = \
                    0.5 * (1 + math.cos(math.pi * global_step / max_steps))
                lr_factor = \
                    (eta_min + (base_lr - eta_min) * progress) / base_lr
            return lr_factor
        return _cosine_decay_scheduler

    def _exclude_from_wt_decay(
        self,
        named_params,
        weight_decay,
        skip_list=['bias', 'bn']
    ):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {
                'params': params,
                'weight_decay': weight_decay
            },
            {
                'params': excluded_params,
                'weight_decay': 0.
            },
        ]

    def _filter_params(self):
        # Exclude biases and bn
        if self.hparams.optimizer == "lars":
            params = self._exclude_from_wt_decay(
                self.named_parameters(),
                weight_decay=self.hparams.weight_decay
            )
        else:
            params = self.parameters()
        return params
