import os
import math
import hydra
import torch
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
    cfg.model.scheduler.warmup_steps = \
        cfg.basic.warmup_epochs * steps_per_epoch
    cfg.model.scheduler.max_steps = \
        cfg.trainer.max_epochs * steps_per_epoch

    # Update arguments in backbone
    if cfg.basic.name == "sim_estimator" and cfg.model.mlp.norm == "gn":
        rounded_power = math.ceil(math.log2(cfg.datamodule.basic.num_classes))
        cfg.model.mlp.num_groups = 2 ** rounded_power

    # Update arguments in optimizer
    cfg.model.optimizer.lr = eval(cfg.model.optimizer.lr)

    # Update arguments in trainer
    cfg.trainer.sync_batchnorm = True if cfg.basic.num_gpus > 1 else False
    available_batches = data_module.batch_size * cfg.basic.num_gpus
    cfg.trainer.accumulate_grad_batches = \
        int(cfg.model.basic.eff_batch_size/available_batches)
    if cfg.trainer.accumulate_grad_batches < 1:
        raise ValueError(
            "[Error] Effective batch"
            f"({cfg.model.basic.eff_batch_size}) size is too small!"
        )

    # Check resume & pretrain
    cfg.trainer.resume_from_checkpoint = \
        _check_ckpt(cfg.trainer.resume_from_checkpoint, cfg.basic.cwd)
    if cfg.basic.stage in ["linear_eval", "self_train"]:
        ckpt_path_list = cfg.model.basic.ckpt_path.split("/")
        pretrained_version = ckpt_path_list[ckpt_path_list.index("log")+3]
        cfg.model.basic.ckpt_path = _check_ckpt(
            cfg.basic.pretrained, cfg.basic.cwd, accept_none=False)
        print(f"[ INFO ] Using weights from {cfg.model.basic.ckpt_path}")
        # Update log dir if in linear_eval mode
        for name, logger in cfg.logger.items():
            if name == "wandb_logger":
                version_str_list = pretrained_version.split('_')
                postfix_start = version_str_list.index(data_module.name) + 1
                postfix = '_'.join(version_str_list[postfix_start:])
                cfg.basic.log_postfix = postfix + '_' + cfg.basic.stage
            else:
                logger.version = \
                    os.path.join(pretrained_version, cfg.basic.stage)

    return cfg


def instantiate_list(class_cfg: DictConfig) -> list:
    instance_list = list()
    if class_cfg is not None:
        for _, config in class_cfg.items():
            instance_list.append(hydra.utils.instantiate(config))

    return instance_list


def _check_config(cfg):
    if torch.cuda.device_count() < cfg.basic.num_gpus:
        raise ValueError(
            f"[Error] GPU amount required({cfg.basic.num_gpus})"
            f"exceed maximum({torch.cuda.device_count()})!"
        )

    if (
        cfg.basic.stage == "self_train"
        and "linear_eval" not in cfg.basic.pretrained
    ):
        raise ValueError(
            "[Error] Should use checkpoint from linear evaluation, "
            f"but not {cfg.basic.pretrained}"
        )

    return


def _check_ckpt(path, cwd, accept_none=True):
    need_cwd = (path is not None) and (not os.path.isfile(path))
    # Relative path may be wrong since cwd changed
    if need_cwd:
        path = os.path.join(cwd, path)
        if not os.path.isfile(path):
            # Path is still wrong after add cwd path
            raise ValueError(f"[Error] Wrong ckpt path: {path}!")
    # Linear_eval need a pretrained weight path
    elif not accept_none:
        raise ValueError(f"[Error] Wrong ckpt path: {path}!")

    return path
