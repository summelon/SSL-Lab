import os
import math
import torch
from PIL import Image

from . import dataset_reader as dsreader


READER = {
        # NOTE: Caltech seed is fixed as 1234
        'caltech101': dsreader.caltech101_reader,
        'stanford_dogs': dsreader.stanford_dogs_reader,
        'food101': dsreader.food101_reader,
        'diabetic_250k': dsreader.diabetic_250k_reader,
        'diabetic_btgraham': dsreader.diabetic_btgraham_reader,
        'food11': dsreader.food11_reader,
        'imagenette': dsreader.imagenette_reader,
        'imagewoof': dsreader.imagewoof_reader,
        'imagewang': dsreader.imagewang_reader,
}


class TorchDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            name,
            trans_func=None,
            is_train=True,
            base_dir='/home/shortcake7/data/',
            **kwargs,
            ):
        # NOTE transformation for test dataset is required
        self.trans_func = trans_func

        name, ratio_num = self._parse_name(name)
        if not is_train:
            ratio_num = 1.0
        print(f"[ INFO ] Ratio number in {'train' if is_train else 'val'}"
              f" is {ratio_num:.2f}")
        data_dir = os.path.join(base_dir, name)
        full_img, full_lbl, self.classes = READER[name](
                is_train=is_train, data_dir=data_dir, **kwargs)
        assert len(full_img) > 0, f"[ Error ] path {data_dir} is wrong!"
        self.class_counts = self._count_cls(full_img, full_lbl, ratio_num)
        self.file_list, self.label_list = \
            self._split_train_set(full_img, full_lbl)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image_path = self.file_list[index]
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[index]
        # class_name = os.path.basename(os.path.dirname(image_path))
        # label = self.class_names.index(class_name)
        if self.trans_func is not None:
            image = self.trans_func(image)
        return image, label

    def _parse_name(self, name):
        parsed_name = name.split(':')
        name = parsed_name[0]

        if len(parsed_name) == 1:
            ratio = 1.0
        elif len(parsed_name) == 2:
            ratio = float(parsed_name[1])
        else:
            raise ValueError("[ Error ] Received an illegal name format")
        return name, ratio

    def _count_cls(self, file_list, label_list, ratio_num):
        from collections import Counter
        counter = Counter()
        for lbl_idx in label_list:
            counter.update([self.classes[lbl_idx]])
        ratio = (ratio_num / len(file_list)) if ratio_num > 1.0 else ratio_num
        # At least one image per class
        assert ratio >= len(self.classes) / len(file_list)
        assert ratio <= 1.0
        counts = [math.ceil(counter[cls]*ratio) for cls in self.classes]
        return counts

    def _split_train_set(self, full_image, full_label):
        # NOTE: Need check for high-level fixed seed
        partial_img_list, partial_lbl_list = list(), list()
        counter = dict((cls, val) for cls, val
                       in zip(self.classes, self.class_counts))
        for img, lbl in zip(full_image, full_label):
            cls = self.classes[lbl]
            if counter[cls] > 0:
                partial_img_list.append(img)
                partial_lbl_list.append(lbl)
                counter[cls] -= 1
        return partial_img_list, partial_lbl_list

    def visualize(self, sample_num):
        import random
        import matplotlib.pyplot as plt

        size = math.sqrt(sample_num)
        assert (size % 1) == 0, "sample_num should be sqrtable as an int."

        idx_list = random.sample(range(len(self)), sample_num)
        size = int(size)
        fig, axes = plt.subplots(size, size, constrained_layout=True)
        for x in range(size):
            for y in range(size):
                image, label = self.__getitem__(idx_list[x*size+y])
                image = self._denormalize(image)
                axes[x][y].imshow(image)
                axes[x][y].set_title(self.classes[label])
                axes[x][y].axis('off')

        plt.savefig('example.png')
        return

    def _denormalize(self, image):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = image.permute(1, 2, 0).numpy()
        image = (image * std + mean) * 255
        image = image.astype('uint8')
        return image


def test_aug():
    import sys
    sys.path.append('/home/project/ssl/moco')
    import moco.loader
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    return transforms.Compose(augmentation)


def main():
    import numpy as np
    np.random.seed(666)
    name = 'diabetic'
    file_dir = './meta'

    # NOTE check random seed in self-split dataset
    # Train
    train_dataset = TorchDataset(name, test_aug(), is_train=True)
    train_list = [
            os.path.join(*(f.split('/')[-2:]))
            for f in train_dataset.file_list]
    train_label_list = [label_idx for label_idx in train_dataset.label_list]

    with open(os.path.join(file_dir, name+'_train.txt'), 'w') as f:
        for path in train_list:
            f.write(path+'\n')
    with open(os.path.join(file_dir, name+'_train_labeled.txt'), 'w') as f:
        for path, label in zip(train_list, train_label_list):
            f.write(path+' '+str(label)+'\n')

    # Test
    test_dataset = TorchDataset(name, None, is_train=False)
    test_list = [
            os.path.join(*(f.split('/')[-2:]))
            for f in test_dataset.file_list]
    test_label_list = [label_idx for label_idx in test_dataset.label_list]

    with open(os.path.join(file_dir, name+'_test.txt'), 'w') as f:
        for path in test_list:
            f.write(path+'\n')
    with open(os.path.join(file_dir, name+'_test_labeled.txt'), 'w') as f:
        for path, label in zip(test_list, test_label_list):
            f.write(path+' '+str(label)+'\n')

    return


if __name__ == "__main__":
    main()
