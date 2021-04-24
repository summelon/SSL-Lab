from typing import Optional

import re
import yaml
from pytorch_lightning import (
    LightningDataModule,
)


def get_config(
    yaml_file_path: str,
    data_module: LightningDataModule,
    gpu_num: int = 4,
    # TODO: parameter `backbone` maybe None in Linear_eval
    backbone: Optional[str] = None,
):
    # Read model arguments
    yaml_loader = _get_yaml_loader()
    with open(yaml_file_path, 'r') as yaml_file:
        args = yaml.load(yaml_file, Loader=yaml_loader)
    # Select specific backbone args
    args = args['base'] if args.get(backbone) is None else args[backbone]

    # Dataset info
    if data_module.name == 'cifar10':
        train_samples = data_module.num_samples
        args['model']['num_classes'] = data_module.num_classes
        args["model"].update(args["cifar"])
    else:
        train_samples = len(data_module.train_set)
        args['model']['num_classes'] = len(data_module.train_set.classes)

    # Batch size & update interval
    if 'linear_eval.yml' in yaml_file_path:
        # TODO Modify eff_batch_size to full-mem-usage fasion
        args['model']['eff_batch_size'] = data_module.batch_size * gpu_num
    args['trainer']['accumulate_grad_batches'] = \
        int(args['model']['eff_batch_size'] / data_module.batch_size / gpu_num)
    if args['trainer']['accumulate_grad_batches'] < 1.0:
        raise ValueError("[Error] Effective batch size is too small!")

    # Calculate steps
    args['model']['warm_up_steps'] = args['basic']['warm_up_epochs'] \
        * train_samples // args['model']['eff_batch_size']
    args['model']['max_steps'] = args['trainer']['max_epochs'] \
        * train_samples // args['model']['eff_batch_size']


    # Configs
    model_config = dict(
        backbone=backbone,
        **args['model'],
    )
    trainer_config = dict(
        gpus=gpu_num,
        sync_batchnorm=True if gpu_num > 1 else False,
        **args['trainer']
        # Callbacks
    )
    config = dict(
        model_config=model_config,
        trainer_config=trainer_config,
    )

    return config


def _get_yaml_loader():
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.')
    )
    return loader


def main():
    import dlhelper
    data_module = dlhelper.dataset.PLDataModule('food11', '/home/data/', 32)
    data_module.setup()
    yaml_file_path = './simsiam.yml'
    config = get_config(
        yaml_file_path=yaml_file_path,
        data_module=data_module,
        gpu_num=4,
        backbone='resnet18',
    )
    print(config)


if __name__ == "__main__":
    main()
