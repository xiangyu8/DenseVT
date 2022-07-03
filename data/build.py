# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from torchvision.datasets import SVHN, CIFAR10, CIFAR100
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import str_to_interp_mode

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler

from .dct_utils import cvtransforms as transforms_cv
from .dct_utils.dctparameters import mean_8, std_8

def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET  == 'imagenet':
        if config.DATA.ZIP_MODE:
            prefix = 'train' if is_train else 'val'
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            prefix = 'train' if is_train else 'val'
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000 
    elif config.DATA.DATASET == 'imagenet-100':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 100
    elif config.DATA.DATASET == 'tiny-imagenet-200':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 200
    elif config.DATA.DATASET == 'flowers102':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 102
    elif config.DATA.DATASET in ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]:
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 345
    elif config.DATA.DATASET in ['cifar-10', 'cifar-100', 'svhn']:
        if config.DATA.DATASET == 'cifar-10':
            #root = os.path.join(config.DATA.DATA_PATH)
            #dataset = datasets.ImageFolder(root, transform=transform)
            dataset = CIFAR10(config.DATA.DATA_PATH, train=is_train, transform=transform)
            nb_classes = 10
        elif config.DATA.DATASET == 'cifar-100':
            dataset = CIFAR100(config.DATA.DATA_PATH, train=is_train, transform=transform)
            nb_classes = 100
        else:
            split = "train" if is_train else "test"
            dataset = SVHN(config.DATA.DATA_PATH, split=split, transform=transform)
            nb_classes = 10
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if config.DATA.DCT_STATUS:
        if config.DATA.DCT_WINDOW ==8:
            normalize_dct_param = dict(y_mean = mean_8, y_std = std_8)
        else:
            raise ValueError("Unsupported Window Size for normalization!")

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        if config.DATA.DCT_STATUS:
            if config.AUG.REPROB > 0:
                transform.transforms = transform.transforms[:-3]
            else:
                transform.transforms = transform.transforms[:-2]
            transform.transforms.append(transforms_cv.ToImageCV())
            transform.transforms.append(transforms_cv.GetDCT(config.DATA.DCT_WINDOW))
            transform.transforms.append(transforms_cv.UpScaleDCT())
            transform.transforms.append(transforms_cv.ToTensorDCT())
            transform.transforms.append(transforms_cv.SubsetDCT(channels = config.DATA.CHANNELS))
            transform.transforms.append(transforms_cv.Aggregate())
            transform.transforms.append(transforms_cv.NormalizeDCT(**normalize_dct_param, channels = config.DATA.CHANNELS))	

        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=str_to_interp_mode(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=str_to_interp_mode(config.DATA.INTERPOLATION))
            )

    if config.DATA.DCT_STATUS:
        t.append(transforms_cv.ToImageCV())
        t.append(transforms_cv.GetDCT(config.DATA.DCT_WINDOW))
        t.append(transforms_cv.UpScaleDCT())
        t.append(transforms_cv.ToTensorDCT())
        t.append(transforms_cv.SubsetDCT(channels = config.DATA.CHANNELS))
        t.append(transforms_cv.Aggregate())
        t.append(transforms_cv.NormalizeDCT(**normalize_dct_param, channels = config.DATA.CHANNELS))	
    else:
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
