import hydra
import warnings
import logging
from omegaconf import DictConfig

warnings.simplefilter("ignore")
logging.getLogger("lightning").setLevel(logging.INFO)

# TODO Checkpoint will save in the current working dir
#       This is a problem in hydra, since it change the cwd


def run(cfg: DictConfig) -> int:
    import pytorch_lightning as pl

    from utils.config_utils import (
        complete_config,
        instantiate_list,
    )

    # Deterministic
    pl.seed_everything(cfg.basic.seed)

    # DataModule
    data_module: pl.LightningDataModule = _prepare_data_module(cfg)
    cfg: DictConfig = complete_config(cfg, data_module)

    # Prepare model
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        _recursive_=False,
    )

    # Callbacks & logger
    callbacks: list = instantiate_list(cfg.callbacks)
    logger: list = instantiate_list(cfg.logger)

    # Prepare trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    # Start Fitting/Linear_eval
    trainer.fit(model, datamodule=data_module)

    # Test in CIFAR10 dataset
    if cfg.basic.stage in ["linear_eval", "self_train", "supervised"]:
        trainer.test(model, datamodule=data_module)

    return model.global_rank


def _prepare_data_module(cfg: DictConfig):
    from dataset.data_module import PLDataModule
    from dataset.ssl_augmentation import (
        SSLTrainTransform,
        SSLEvalTransform
    )

    cfg.datamodule.data_module.num_workers: int = \
        eval(cfg.datamodule.data_module.num_workers)
    data_module: PLDataModule = \
        hydra.utils.instantiate(cfg.datamodule.data_module)

    data_module.train_transforms: SSLTrainTransform = \
        hydra.utils.instantiate(cfg.transform.train_transforms)
    data_module.val_transforms: SSLEvalTransform = \
        hydra.utils.instantiate(cfg.transform.val_transforms)
    data_module.test_transforms: SSLEvalTransform = \
        hydra.utils.instantiate(cfg.transform.val_transforms)

    data_module.setup()

    return data_module


@hydra.main(config_path="configs/common", config_name="base")
def main(cfg: DictConfig) -> None:
    import time
    import datetime

    tic = int(time.time())

    global_rank = run(cfg)

    toc = int(time.time())
    if global_rank == 0:
        logging.info(f"Total time: {datetime.timedelta(seconds=toc-tic)}")

    return


if __name__ == "__main__":
    main()
