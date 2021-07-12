import random
from typing import Optional
from torchvision import transforms
from PIL import ImageFilter, ImageOps, Image


class SSLTrainTransform(object):
    def __init__(
        self,
        # Augmentation options
        gaussian_blur_prob: float,
        solarize_prob: float,
        jitter_strength: float,
        global_asymmetric_trans: bool,
        # Global
        num_global_crops: int = 2,
        global_input_size: int = 224,
        global_scale: list = (0.08, 1.0),
        # Local
        num_local_crops: int = 0,
        local_input_size: Optional[int] = None,
        local_scale: Optional[list] = None,
        # Normalization fn
        normalize=None,
        # Less augmentation for supervised
        supervised: bool = False,
    ) -> None:
        """
        crop_scale is (0.08, 1.0) by default in trochvision

        Args:
            global_input_size(int): default 224 in normal dataset
            global_scale(list): (0.14, 1.0)

            local_input_size(int): default 96 in normal dataset
            local_scale(list): (0.05, 0.14)
        """
        if num_local_crops > 0:
            assert local_input_size is not None or local_scale is not None, \
                "[Error] Local_input_size and local_scale is required"
        # Global trans are two-way asymmetric
        assert num_global_crops == 2, "[Error] Only support 2 global crops"

        self.supervised = supervised
        self.color_jitter_fn = self._get_color_jitter(jitter_strength)
        self.final_trans = self._get_final_trans(normalize)
        # self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops

        self.global_trans = self._get_global_trans(
            input_size=global_input_size,
            crop_scale=global_scale,
            gaussian_blur_prob=gaussian_blur_prob,
            solarize_prob=solarize_prob,
            asymmetric_trans=global_asymmetric_trans,
        )
        self.local_trans = self._get_local_trans(
            input_size=local_input_size,
            crop_scale=local_scale,
            num_crops=num_local_crops,
        )
        self.online_trans = self._get_online_trans(global_input_size)
        return

    def __call__(self, sample):
        augmentations = [self.online_trans(sample)]
        if not self.supervised:
            # Global views
            for g_trans in self.global_trans:
                augmentations.append(g_trans(sample))
            # Local views if local trnas are defined
            if self.local_trans is not None:
                for _ in range(self.num_local_crops):
                    augmentations.append(self.local_trans(sample))
            # Online views for linear eval
        return augmentations

    def _get_global_trans(
        self,
        input_size: int,
        crop_scale: list,
        gaussian_blur_prob: float,
        solarize_prob: float,
        asymmetric_trans: bool,
    ) -> list:
        global_trans_0 = self._get_ssl_trans(
            input_size=input_size,
            crop_scale=crop_scale,
            gaussian_blur_prob=gaussian_blur_prob,
            solarize_prob=solarize_prob,
        )
        if asymmetric_trans:
            global_trans_1 = self._get_ssl_trans(
                input_size=input_size,
                crop_scale=crop_scale,
                gaussian_blur_prob=0.1,
                solarize_prob=0.2,
            )
        else:
            global_trans_1 = global_trans_0

        return (global_trans_0, global_trans_1)

    def _get_local_trans(self, input_size, crop_scale, num_crops):
        if num_crops != 0:
            local_trans = self._get_ssl_trans(
                input_size=input_size,
                crop_scale=crop_scale,
                gaussian_blur_prob=0.5,
                solarize_prob=0.0,
            )
        else:
            local_trans = None
        return local_trans

    def _get_ssl_trans(
        self,
        input_size: int,
        crop_scale: list,
        gaussian_blur_prob: float,
        solarize_prob: float,
    ):
        # Basic part
        basic_trans = [
            transforms.RandomResizedCrop(
                size=input_size,
                scale=crop_scale,
                interpolation=Image.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter_fn], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        # Additive/asymmetric part
        asym_trans = list()
        if gaussian_blur_prob > 0:
            asym_trans.append(GaussianBlur(gaussian_blur_prob))
        if solarize_prob > 0:
            asym_trans.append(Solarize(solarize_prob))
        # Return combination
        return transforms.Compose(basic_trans+asym_trans+self.final_trans)

    def _get_online_trans(self, input_size):
        """Weak augmentation for Linear_eval"""
        basic_trans = [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        return transforms.Compose(basic_trans+self.final_trans)

    def _get_final_trans(self, normalize_fn):
        if normalize_fn is None:
            final_transform = [transforms.ToTensor()]
        else:
            final_transform = [transforms.ToTensor(), normalize_fn]
        return final_transform

    def _get_color_jitter(self, strength=0.5):
        color_jitter_fn = transforms.ColorJitter(
            0.8 * strength,
            0.8 * strength,
            0.4 * strength,
            0.2 * strength,
        )
        return color_jitter_fn


class SSLEvalTransform():
    """
    Transforms for SimCLR
    Transform::
        Resize(input_height + 10, interpolation=3)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()
    Example::
        from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform
        transform = SimCLREvalDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self,
        input_size: int = 224,
        normalize=None,
    ):
        resize_size = int(input_size + 0.1 * input_size)
        basic_trans = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
        ]
        if normalize is None:
            final_trans = [transforms.ToTensor()]
        else:
            final_trans = [transforms.ToTensor(), normalize]

        self.online_transform = transforms.Compose(basic_trans+final_trans)

    def __call__(self, sample):
        # TODO: for consistency with the code for now, modify in the future
        return [self.online_transform(sample)]


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarize(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
