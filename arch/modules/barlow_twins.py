import lightly
from omegaconf import DictConfig

from .simsiam import SimSiamModel


class BarlowTwinsModel(SimSiamModel):
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
        # TODO: Add hyper-parameter `lambd` according to GitHub repo
        # TODO: Add hyper-parameter `scale-loss` according to GitHub repo
        # Ref: https://github.com/facebookresearch/barlowtwins#barlow-twins-training

        self.criterion = lightly.loss.BarlowTwinsLoss()

    def training_step(self, batch, batch_idx):
        # (x0, x1), _, _ = batch
        # (w/o aug, Aug0, Aug1), label
        (_, x0, x1), _ = batch
        (y0, _), (y1, _) = self.online_network(x0, x1)
        loss = self.criterion(y0, y1)
        # TODO: modify to more clean method
        self.outputs = y0
        self.log(
            'repr_loss', loss, prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True
        )
        return loss
