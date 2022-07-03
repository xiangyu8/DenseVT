# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin import SwinTransformer
from .t2t import T2t_vit_14
from .resnet import ResNet50
from .vit import deit_small
from .focalvit import focal_tiny 

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE)
    elif model_type == 'swin_dct':
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.DATA.DCT_WINDOW,
            in_chans=config.DATA.CHANNELS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE,
            dct_status=True,
            attn_status = False)
    elif model_type == 'swin_dct_attn':
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.DATA.DCT_WINDOW,
            in_chans=config.DATA.CHANNELS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE,
            dct_status=True,
            attn_status = True)
    elif model_type == "t2t":
        model = T2t_vit_14(
            img_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            sample_size=config.TRAIN.SAMPLE_SIZE,
        )
    elif model_type == 'resnet50':
        model = ResNet50(
            num_classes=config.MODEL.NUM_CLASSES,
            sample_size=config.TRAIN.SAMPLE_SIZE,
        )
    elif model_type == 'resnet50_dct':
        model = ResNetDCT_Upscaled_Static(
            channels = config.DATA.CHANNELS,
            num_classes=config.MODEL.NUM_CLASSES,
        )
    elif model_type == "focalvit": # focal tiny
        model = focal_tiny(
            num_classes=config.MODEL.NUM_CLASSES,
        )
    elif model_type == "focalvit_dct": # deit_small_attn
        model = focal_tiny(
            num_classes=config.MODEL.NUM_CLASSES,
            dct_status = True,
            attn_status = False,
            in_chans = config.DATA.CHANNELS,
            patch_size=config.DATA.DCT_WINDOW,
            img_size = config.DATA.IMG_SIZE,
        )
    elif model_type == "focalvit_dct_attn": # deit_small_attn
        model = focal_tiny(
            num_classes=config.MODEL.NUM_CLASSES,
            dct_status = True,
            attn_status = True,
            in_chans = config.DATA.CHANNELS,
            patch_size=config.DATA.DCT_WINDOW,
            img_size = config.DATA.IMG_SIZE,
        )
    elif model_type == "deit":
        model = deit_small(
            num_classes=config.MODEL.NUM_CLASSES,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
