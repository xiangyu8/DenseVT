# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .vit import deit_small, deit_small_mix
from .backbone import ResNet10_dct_bridge, ResNet50_dct_bridge, ResNet18_dct_bridge

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "deit":
        model = deit_small(
            num_classes=config.MODEL.NUM_CLASSES,
        )
    elif model_type == "deit_dct":
        model = deit_small(
            img_size = [config.DATA.IMG_SIZE//config.DATA.DCT_WINDOW],
            num_classes=config.MODEL.NUM_CLASSES,
            in_chans = config.DATA.CHANNELS,
            dct_status = True,
        )
    elif model_type == "deit_dct_mix":
        model = deit_small_mix(
            img_size = [config.DATA.IMG_SIZE//config.DATA.DCT_WINDOW],
            num_classes=config.MODEL.NUM_CLASSES,
            in_chans = config.DATA.CHANNELS,
            dct_status = True,
        )
    elif model_type == "resnet10_dct":
        model = ResNet10_dct_bridge(num_classes = config.MODEL.NUM_CLASSES, pre_indim = config.DATA.CHANNELS)
    elif model_type == "resnet18_dct":
        model = ResNet18_dct_bridge(num_classes = config.MODEL.NUM_CLASSES, pre_indim = config.DATA.CHANNELS)
    elif model_type == "resnet50_dct":
        model = ResNet50_dct_bridge(num_classes = config.MODEL.NUM_CLASSES,
        pre_indim = config.DATA.CHANNELS)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
