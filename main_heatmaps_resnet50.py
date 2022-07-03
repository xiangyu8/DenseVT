# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data.build import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import (
    load_checkpoint, 
    save_checkpoint, 
    save_checkpoint_best,
    get_grad_norm, 
    auto_resume_helper, 
    reduce_tensor
)


from pytorch_grad_cam import GradCAM

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
   # import torch.cuda.amp as amp
except ImportError:
    amp = None

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument("--sample_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--use_multiscale", action='store_true')
    parser.add_argument("--ape", action="store_true", help="using absolute position embedding")
    parser.add_argument("--rpe", action="store_false", help="using relative position embedding")
    parser.add_argument("--use_normal", action="store_true")
    parser.add_argument("--use_abs", action="store_true")
    parser.add_argument("--ssl_warmup_epochs", type=int, default=20)
    parser.add_argument("--total_epochs", type=int, default=100)
    parser.add_argument("--dct_status", action='store_true')
    parser.add_argument("--channels", type=int, default=0)
    parser.add_argument("--dct_window", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def _weight_decay(init_weight, epoch, warmup_epochs=20, total_epoch=300):
    if epoch <= warmup_epochs:
        cur_weight = min(init_weight / warmup_epochs * epoch, init_weight)
    else:
        cur_weight = init_weight * (1.0 - (epoch - warmup_epochs)/(total_epoch - warmup_epochs))
    return cur_weight

def main():
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    config.defrost()
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())


    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    # dataset_train, data_loader_train, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))
    
    criterion_sup = torch.nn.CrossEntropyLoss()
    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        print("resume: ", config.MODEL.RESUME)
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict = False)
        print(model)
        acc1, acc5, loss = validate(config, data_loader_train, model, logger)
        if config.EVAL_MODE:
            return

def validate(config, data_loader, model, logger):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    num_imgs = 0

    avg_channel_heatmaps = np.zeros((3*config.DATA.DCT_WINDOW**2, config.DATA.IMG_SIZE//config.DATA.DCT_WINDOW, config.DATA.IMG_SIZE//config.DATA.DCT_WINDOW ))

    for idx, (images, target) in enumerate(data_loader):
        if idx > 0:
            break
        print(idx, len(data_loader))
        num_imgs = images.shape[0] + num_imgs

        images = images.cuda(non_blocking=True)

        for i in range(images.shape[1]):
            input_channels = torch.zeros(images.shape).cuda()
            input_channels[:,i] = images[:,i]
            target_layers = [model.trunk[4]]
            
            cam = GradCAM(model = model, target_layers = target_layers, use_cuda = True)
            target_category = None
            grayscale_cam = cam(input_tensor = input_channels, )
            grayscale_cam = np.sum(grayscale_cam, 0)
            avg_channel_heatmaps[i] = avg_channel_heatmaps[i] + grayscale_cam

    avg_channel_heatmaps /= num_imgs   

    out_file = config.OUTPUT +"/avg_channel_heatmaps_rlt_avgbatch"
    np.save(out_file, avg_channel_heatmaps)

    return None, None, None


if __name__ == '__main__':
    main()
