import os
import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule


def complete_config(
        cfg: DictConfig,
        data_module: LightningDataModule
) -> DictConfig:

    # Check some error may be omit
    _check_config(cfg)

    # Update arguments in scheduler
    if data_module.name == 'cifar10':
        cfg.datamodule.basic.train_samples = data_module.num_samples
        cfg.datamodule.basic.num_classes = data_module.num_classes
    else:
        cfg.datamodule.basic.train_samples = len(data_module.train_set)
        cfg.datamodule.basic.num_classes = len(data_module.train_set.classes)

    steps_per_epoch = \
        cfg.datamodule.basic.train_samples // cfg.model.basic.eff_batch_size
    cfg.model.scheduler = dict(
        warm_up_steps=cfg.basic.warm_up_epochs*steps_per_epoch,
        max_steps=cfg.trainer.max_epochs*steps_per_epoch,
    )

    # Update arguments in optimizer
    cfg.model.optimizer.lr = eval(cfg.model.optimizer.lr)

    # Update arguments in trainer
    cfg.trainer.sync_batchnorm = True if cfg.basic.num_gpus > 1 else False
    available_batches = data_module.batch_size * cfg.basic.num_gpus
    cfg.trainer.accumulate_grad_batches = \
        int(cfg.model.basic.eff_batch_size/available_batches)
    if cfg.trainer.accumulate_grad_batches < 1:
        raise ValueError("[Error] Effective batch size is too small!")

    # Check resume & pretrain
    cfg.trainer.resume_from_checkpoint = \
        _check_ckpt(cfg.trainer.resume_from_checkpoint, cfg.basic.cwd)
    if cfg.basic.stage == "linear_eval":
        cfg.model.basic.ckpt_path = _check_ckpt(
            cfg.basic.pretrained, cfg.basic.cwd, accept_none=False)
        print(f"[ INFO ] Using weights from {cfg.model.basic.ckpt_path}")
        # Update log dir if in linear_eval mode
        pretrained_version = cfg.model.basic.ckpt_path.split('/')[-3]
        cfg.logger.tensorboard_logger.version = \
            os.path.join(pretrained_version, "linear_eval")

    return cfg


def instantiate_list(class_cfg: DictConfig) -> list:
    instance_list = list()
    if class_cfg is not None:
        for _, config in class_cfg.items():
            instance_list.append(hydra.utils.instantiate(config))

    return instance_list


def _check_config(cfg):
    pass


def _check_ckpt(path, cwd, accept_none=True):
    need_cwd = (path is not None) and (not os.path.isfile(path))
    if need_cwd:
        path = os.path.join(cwd, path)
        if not os.path.isfile(path):
            raise ValueError(f"[Error] Wrong ckpt path: {path}!")
    elif not accept_none:
        raise ValueError(f"[Error] Wrong ckpt path: {path}!")

    return path
