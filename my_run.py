import hydra
import warnings
warnings.simplefilter("ignore")


def run(cfg):
    # Official module
    import os
    import time
    import datetime
    import pytorch_lightning as pl

    # Current dir
    import callbacks.custom_callbacks as custom_callbacks
    from configs.get_config import get_config
    from arguments import select_model

    pl.seed_everything(args["seed"])

    tic = int(time.time())
    # # Select specific SSL module
    # LightningModule = select_model(args['model'], args['stage'])
    # # Hyperparameters
    # yaml_file = 'linear_eval' if args['stage'] == 'test' else args['model']
    # yaml_file_path = os.path.join('./configs', yaml_file+'.yml')

    # DataModule
    data_module = _prepare_data_module(args)
    # Logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), 'log', args['model']),
        name=args['backbone'],
        version=args["version_name"],
    )
    # Module-specifi config
    config = get_config(
        yaml_file_path=yaml_file_path,
        data_module=data_module,
        gpu_num=args['gpu'],
        backbone=args['backbone'],
    )
    # Use pretrain weights
    if args['stage'] == 'test':
        config["model_config"]["ckpt_path"] = args["pretrained"]
    # Prepare model
    model = LightningModule(**config['model_config'])
    # Callbacks
    if args['stage'] == 'fit':
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
        online_linear = custom_callbacks.OnlineLinear(
            num_features=model.online_network.num_features,
            num_classes=model.hparams.num_classes,
            dataset=args['dataset']
        )
        callbacks = [
                lr_monitor,
                online_linear,
                custom_callbacks.CheckCollapse(),
        ]
    else:
        callbacks = list()
    # Prepare trainer
    trainer = pl.Trainer(
        **config['trainer_config'],
        logger=tb_logger,
        resume_from_checkpoint=args['resume'],
        callbacks=callbacks,
    )
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

    data_module: LightningDataModule = \
        hydra.utils.instantiate(cfg.datamodule.data_module)

    data_module.train_transforms: SimCLRTrainDataTransform = \
        hydra.utils.instantiate(cfg.datamodule.train_transforms)
    data_module.val_transforms: SimCLREvalDataTransform = \
        hydra.utils.instantiate(cfg.datamodule.val_transforms)

    data_module.setup()

    return


if __name__ == "__main__":
    from arguments import param_loader
    args = param_loader()
    run(args)
