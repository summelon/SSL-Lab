import warnings
warnings.simplefilter("ignore")


def main(args):
    # Official module
    import os
    import time
    import datetime
    import pytorch_lightning as pl

    # Current dir
    import callbacks.custom_callbacks as custom_callbacks
    from configs.get_config import get_config
    from arguments import select_model

    tic = int(time.time())
    # Select specific SSL module
    LightningModule = select_model(args['model'], args['stage'])
    # Hyperparameters
    yaml_file = 'linear_eval' if args['stage'] == 'test' else args['model']
    yaml_file_path = os.path.join('./configs', yaml_file+'.yml')

    # DataModule
    data_module = _prepare_data_module(args)
    # Logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), 'log', args['model']),
        name=args['backbone'],
        version=args['dataset'],
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


def _prepare_data_module(args):
    import dlhelper
    from pl_bolts.models.self_supervised.simclr import (
        SimCLRTrainDataTransform,
        SimCLREvalDataTransform,
    )
    dataset_name = args['dataset']
    # Device bs for real acceptable batch size 1080Ti & TiTan-xp
    if args['backbone'] == 'resnet18':
        device_batch_size = 128 if args['model'] != 'barlow_twins' else 64
    elif args['backbone'] == 'resnet50':
        device_batch_size = 32

    # Prepare data module
    if dataset_name == 'cifar10':
        from pl_bolts.datamodules import CIFAR10DataModule
        from pl_bolts.transforms.dataset_normalizations import (
            cifar10_normalization
        )
        input_size = 32
        data_module = CIFAR10DataModule(
            data_dir=args['base_dir'],
            num_workers=8,
            batch_size=128 if args["backbone"] == "resnet50" else 512,
            shuffle=True,
            pin_memory=True,
            # TODO Check if fixed split
            val_split=5000,
        )
        data_module.train_transforms = SimCLRTrainDataTransform(
            input_size,
            gaussian_blur=False,
            jitter_strength=0.5,
            normalize=cifar10_normalization(),
        )
        data_module.val_transforms = SimCLREvalDataTransform(
            input_size,
            gaussian_blur=False,
            jitter_strength=0.5,
            normalize=cifar10_normalization(),
        )
    else:
        input_size = 224
        data_module = dlhelper.dataset.PLDataModule(
            dataset_name=dataset_name,
            base_dir=args['base_dir'],
            batch_size=device_batch_size,
            num_workers=4*args['gpu'],
        )
        data_module.train_transforms = SimCLRTrainDataTransform(input_size)
        data_module.val_transforms = SimCLREvalDataTransform(input_size)
    data_module.setup()

    return data_module


if __name__ == "__main__":
    from arguments import param_loader
    args = param_loader()
    main(args)
