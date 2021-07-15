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

    def _load_state_dict_to_specific_part(self, network, state_dict):
        # Clean up weight
        pretrained_keys = list(state_dict["state_dict"].keys())
        for k in pretrained_keys:
            if "conv1.weight" not in k:
                _ = state_dict["state_dict"].pop(k)
            else:
                break
        dict_zip = zip(
            state_dict["state_dict"].items(),
            network.state_dict().items()
        )
        match_dict = {}
        for (s_k, s_v), (m_k, m_v) in dict_zip:
            if (m_k in s_k) and (s_v.shape == m_v.shape):
                match_dict[m_k] = s_v
        msg = network.load_state_dict(match_dict, strict=False)
        print(f"[ INFO ] Missing keys: {msg.missing_keys}")
        return

    def _multi_crop_forward(
        self,
        images: list,
        network: torch.nn.Module,
        local_forward: bool = False,
        use_projector_feature: bool = False,
    ) -> torch.Tensor:

        g_crops = self.hparams.basic.num_global_crops
        feat_idx = 0 if use_projector_feature else 1

        # Global features
        # features = (proj_features, pred_features)
        features = network(torch.cat(images[:g_crops]))
        # Outputs for collapse checker(local_forward in online net)
        if local_forward:
            self.outputs = features[0]
        features = features[feat_idx]

        # Local features if have local crops
        if local_forward and self.hparams.basic.num_local_crops > 0:
            local_features = network(torch.cat(images[g_crops:]))
            features = torch.cat((features, local_features[feat_idx]))
        return features
