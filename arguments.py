import os
from argparse import ArgumentParser


def param_loader():
    parser = ArgumentParser()
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
    # Dataset
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name"
    )
    parser.add_argument(
        "--base_dir", type=str, default="/dev/shm",
        help="The base directory to where dataset is"
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

    if args.stage == 'test':
        if args.pretrained and os.path.isfile(args.pretrained):
            print(f"[ INFO ] Using weights from {args.pretrained}")
        else:
            raise ValueError("[ Error ] Wrong pretrained file path!")

    return vars(args)


def select_model(model_name: str, stage: str):
    # TODO add return notion
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
