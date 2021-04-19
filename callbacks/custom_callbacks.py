import time
import math
import torch
import logging
import pl_bolts
import pytorch_lightning as pl


class CheckCollapse(pl.callbacks.Callback):
    def __init__(self):
        super(CheckCollapse, self).__init__()
        self.w = 0.9
        self.avg_output_std = 0.0

    def on_train_batch_end(
            self, trainer, pl_module,
            outputs, batch, batch_idx, dl_idx
    ):
        output = pl_module.outputs.detach()
        output = torch.nn.functional.normalize(output, dim=1)
        output_std = torch.std(output, 0).mean()
        self.avg_output_std = \
            self.w * self.avg_output_std + (1 - self.w) * output_std.item()

        # the level of collapse is large if the standard deviation of the l2
        # normalized output is much smaller than 1 / sqrt(dim)
        out_dim = pl_module.online_network.num_ftrs
        collapse_level = \
            max(0., 1 - math.sqrt(out_dim) * self.avg_output_std)
        # Log to tensorboard
        pl_module.log(
            'collapse', collapse_level,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True)
        return


# Adaptively inherit the SSLOnlineEvaluator from pl_bots
class OnlineLinear(pl_bolts.callbacks.SSLOnlineEvaluator):
    def __init__(self, num_features, num_classes, dataset='imagenet'):
        super(OnlineLinear, self).__init__(
            dataset=dataset,
            drop_p=0.0,
            hidden_dim=None,
            z_dim=num_features,
            num_classes=num_classes
        )

    def on_train_batch_end(
            self, trainer, pl_module, outputs,
            batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)
        mlp_loss = torch.nn.functional.cross_entropy(mlp_preds, y)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log metrics
        train_acc = pl.metrics.functional.accuracy(mlp_preds, y)
        pl_module.log(
            'online_train_acc', train_acc,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True)
        pl_module.log(
            'online_train_loss', mlp_loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=False, sync_dist=True)

    def on_validation_batch_end(
            self, trainer, pl_module, outputs,
            batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)
        mlp_loss = torch.nn.functional.cross_entropy(mlp_preds, y)

        # log metrics
        val_acc = pl.metrics.functional.accuracy(mlp_preds, y)
        pl_module.log(
            'online_val_acc', val_acc,
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log(
            'online_val_loss', mlp_loss,
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True, sync_dist=True)
