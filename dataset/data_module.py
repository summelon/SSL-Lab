import torch
import torchvision
import pytorch_lightning as pl
from typing import Optional

from .random_access_dataset import TorchDataset


class PLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        base_dir: str,
        batch_size: int,
        trans_func: dict = dict(),
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.specific_kwargs = kwargs
        self.name = dataset_name
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self._set_trans_func(trans_func)

    def setup(self, stage: Optional[str] = None):
        self.train_set = self._prepare_dataset(
            trans_func=self.train_transforms,
            is_train=True
        )
        self.val_set = self._prepare_dataset(
            trans_func=self.val_transforms,
            is_train=False
        )
        return

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )
        return val_loader

    def test_dataloader(self):
        print("[INFO] Note using validation set for testing")
        val_loader = torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )
        return val_loader

    def _prepare_dataset(self, trans_func, is_train: bool):
        dataset = TorchDataset(
            name=self.name,
            trans_func=trans_func,
            is_train=is_train,
            base_dir=self.base_dir,
            **self.specific_kwargs,
        )
        return dataset

    def _set_trans_func(self, trans_func: dict):
        default_trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ])
        train_trans = trans_func.get('train')
        self.train_transforms = train_trans if train_trans else default_trans
        val_trans = trans_func.get('val')
        self.val_transforms = val_trans if val_trans else default_trans
        return


def main():
    dm = PLDataModule(
        dataset_name='food11',
        base_dir='/home/data/',
        batch_size=128,
    )
    dm.setup()
    for train_data in dm.train_dataloader():
        print(train_data)
        break

    return


if __name__ == "__main__":
    main()
