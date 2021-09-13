import torch
import torchvision


def get_simple_transform(input_size=224):
    simple_trans = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return simple_trans


def get_data_loader(path: str, batch_size: int):
    dataset = torchvision.datasets.ImageFolder(
        root=path,
        transform=get_simple_transform()
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
    )

    return data_loader


def get_imagewang_loader(path: str, batch_size: int):
    dataset = ImageFolderWithPath(
        root=path,
        transform=get_simple_transform()
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
    )

    return data_loader


def get_food11_loader(path: str, batch_size: int, ratio: float = 0.1):
    from dlhelper.dataset import TorchDataset
    dataset = TorchDataset(
        name="food11:" + str(ratio),
        base_dir=path,
        trans_func=get_simple_transform(),
        is_train=True,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
    )

    return data_loader


class ImageFolderWithPath(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root=root, transform=transform)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        file_name = self.imgs[idx][0].split('/')[-1]
        return image, (label, file_name)
