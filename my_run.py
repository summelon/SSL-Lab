import hydra
import warnings
warnings.simplefilter("ignore")


def run(cfg):
    # Official module
    import time
    import datetime
    import pytorch_lightning as pl

    # Current dir
    from utils.config_utils import complete_config

    pl.seed_everything(cfg.basic.seed)

    tic = int(time.time())

    # DataModule
    data_module = _prepare_data_module(cfg)
    cfg = complete_config(cfg, data_module)

    # Prepare model
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        _recursive_=False,
    )

    # Callbacks
    callbacks = list()
    if cfg.callbacks is not None:
        for _, callback_config in cfg.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback_config))

    # Logger
    logger = list()
    if cfg.logger is not None:
        for _, logger_config in cfg.logger.items():
            logger.append(hydra.utils.instantiate(logger_config))

    # Prepare trainer
    # TODO: Check _convert_
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial")

    # Start Fitting/Testing
    trainer.fit(model, datamodule=data_module)

    toc = int(time.time())
    model.print(f"[ INFO ]Total time: {datetime.timedelta(seconds=toc-tic)}")


def _prepare_data_module(cfg):
    from pytorch_lightning import LightningDataModule
    from pl_bolts.models.self_supervised.simclr.transforms import (
        SimCLRTrainDataTransform,
        SimCLREvalDataTransform,
    )

    cfg.datamodule.data_module.num_workers = \
        eval(cfg.datamodule.data_module.num_workers)
    data_module: LightningDataModule = \
        hydra.utils.instantiate(cfg.datamodule.data_module)

    data_module.train_transforms: SimCLRTrainDataTransform = \
        hydra.utils.instantiate(cfg.datamodule.train_transforms)
    data_module.val_transforms: SimCLREvalDataTransform = \
        hydra.utils.instantiate(cfg.datamodule.val_transforms)

    data_module.setup()

    return data_module


if __name__ == "__main__":
    from arguments import param_loader
    args = param_loader()
    run(args)
