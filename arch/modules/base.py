import math
import torch
import pytorch_lightning as pl
from pl_bolts.optimizers.lars import LARS


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
        optimizer = self._select_optimizer()
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

    def _select_optimizer(self):
        # Calculate scaled lr according to effective batch size
        scaled_lr = self.hparams.base_lr * self.hparams.eff_batch_size / 256
        # Select opimizer
        opt_args = dict(
            params=self._filter_params(),
            # params=self.parameters(),
            lr=scaled_lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(**opt_args)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                momentum=self.hparams.momentum, **opt_args)
        elif self.hparams.optimizer == "lars":
            optimizer = LARS(
                **opt_args,
                momentum=0.9,
                trust_coefficient=0.001,
            )
        else:
            raise NotImplementedError("[ Error ] Optimizer is not implemented")
        return optimizer
