import os
from inspect import cleandoc
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from pytorch_lightning import LightningModule


def param_loader():
    parser = ArgumentParser(
        description="SSL_Lab",
        formatter_class=RawTextHelpFormatter,
    )
    # Deterministic
    parser.add_argument(
        "--seed", type=int, default=666,
        help="Random seed number"
    )
    # Train
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=['fit', 'test'],
        help="'fit' or 'test'"
    )
    parser.add_argument(
        "--pretrained", type=str, help="Saved ckpt"
    )
    parser.add_argument(
        "--resume", type=str, help="Path to resume ckpt"
    )
    parser.add_argument(
        "--gpu", type=int, default=4,
        help="number of GPU"
    )
    parser.add_argument(
        "--log_postfix", type=str, default="",
        help="Postfix append to the name of version"
    )
    # Dataset
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name"
    )
    parser.add_argument(
        "--base_dir", type=str, default="/dev/shm",
        help="The base directory to where dataset is"
    )
    parser.add_argument(
        "--device_bs", type=int, required=True,
        help=cleandoc(
            '''
            The batch size on each GPU:
                Cifar10:
                    ResNet18:
                        normal module: 512
                        BarlowTwins: 256
                    ResNet50:
                        normal module: 128
                        BarlowTwins: 64
                224 Dataset:
                    ResNet18:
                        normal module: 128
                    ResNet50:
                        normal module: 32
            '''
        )
    )
    # Model
    parser.add_argument(
        "--backbone", type=str, default="resnet18",
        choices=['resnet18', 'resnet50'],
        help="Backbone model architecture"
    )
    parser.add_argument(
        "--model", type=str,
        choices=['simsiam', 'byol', 'barlow_twins'],
        help="Self-Supervised model"
    )

    # Parse to dict
    args, _ = parser.parse_known_args()
    args.version_name = _get_version_name(args)

    return vars(args)


def _get_version_name(args):
    if args.stage == 'test':
        if args.pretrained and os.path.isfile(args.pretrained):
            print(f"[ INFO ] Using weights from {args.pretrained}")
            version_name = args.pretrained.split('/')[-3]
        else:
            raise ValueError("[ Error ] Wrong pretrained file path!")
    else:
        if args.log_postfix != "":
            version_name = args.dataset + '_' + args.log_postfix
        else:
            version_name = args.dataset

    return version_name


def select_model(model_name: str, stage: str) -> LightningModule:
    if stage == 'fit':
        if model_name == 'simsiam':
            from arch.modules.simsiam import (
                SimSiamModel as LightningModule
            )
        elif model_name == 'byol':
            from arch.modules.byol import (
                BYOLModel as LightningModule
            )
        elif model_name == 'barlow_twins':
            from arch.modules.barlow_twins import (
                BarlowTwins as LightningModule
            )
    elif stage == 'test':
        from arch.modules.linear_eval import (
            LinearEvalModel as LightningModule
        )

    return LightningModule
