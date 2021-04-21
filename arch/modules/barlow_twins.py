import lightly

from .simsiam import SimSiamModel


class BarlowTwins(SimSiamModel):
    def __init__(
        self,
        base_lr: float,
        weight_decay: float,
        momentum: float,
        eff_batch_size: int,
        warm_up_steps: int,
        max_steps: int,
        num_classes: int,
        maxpool1: bool,
        first_conv: bool,
        mlp_config: dict,
        backbone: str = "resnet18",
        optimizer: str = "adam",
    ):
        self.save_hyperparameters()
        super().__init__(**self.hparams)

        self.online_network = self._prepare_model()
        self.criterion = lightly.loss.BarlowTwinsLoss()
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
        y0, y1 = self.online_network(x0, x1)
        loss = self.criterion(y0, y1)
        # TODO: modify to more clean method
        self.outputs = y0
        self.log(
            'repr_loss', loss, prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True
        )
        return loss
