"""
Code for paper:
Unsupervised Voice-Face Representation Learning by
Cross-Modal Prototype Contrast
"""

import sys

sys.path.append("/home/cocoxili/Res/CMPC")

import numpy as np
import os
import argparse
import time
import warnings

import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist
import torch.multiprocessing as mp

from dataloader import *
from dataloader_verfication import *
import utils.logger
from utils import main_utils
from utils import metrics_utils
from utils.torch_utils import *

import models
import criterions
from utils.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from keops_kmeans import *
from criterions import FVMemoryBank
from IPython import embed

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('cfg', help='model directory')

parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument(
    '--dist-url', default='tcp://localhost:15475', type=str, help='url used to set up distributed training'
)
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument(
    '--multiprocessing-distributed',
    action='store_true',
    help='Use multi-processing distributed training to launch '
    'N processes per node, which has N GPUs. This is the '
    'fastest way to use PyTorch for either single node or '
    'multi node data parallel training',
)


def main():
    global args
    args = parser.parse_args()

    global cfg
    cfg = yaml.safe_load(open(args.cfg))

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.seed is not None:
        main_utils.seed_everything(args.seed)
        warnings.warn(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.'
        )

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely ' 'disable data parallelism.')

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        print(args.world_size)
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, cfg)


def main_worker(gpu, ngpus_per_node, args, cfg):
    args.gpu = gpu

    # Setup environment
    args = main_utils.initialize_distributed_backend(args, ngpus_per_node)
    logger, tb_writter, model_dir = main_utils.prep_environment(args, cfg)

    # Define model
    model = models.__dict__[cfg['model']['arch']](**cfg['model']['args'])
    model = model.cuda()

    # Define dataloaders
    dataset = VoxCeleb1(cfg['dataset'])
    sampler = CycledRandomSampler(dataset)

    train_loader = DataLoader(
        dataset,
        batch_size=cfg['dataset']['batch_size'],
        num_workers=cfg['dataset']['num_workers'],
        shuffle=(sampler is None),
        pin_memory=True,
        drop_last=cfg['dataset']['train']['drop_last'],
        sampler=sampler,
    )

    logger.info("\n" + "=" * 20 + "   Train data   " + "=" * 20)
    logger.info(str(train_loader.dataset))

    # Define criterion
    train_criterion = criterions.__dict__[cfg['loss']['name']](**cfg['loss']['args'])
    train_criterion = train_criterion.cuda()

    # Define optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg['optimizer']['lr']['max_lr'],
        weight_decay=cfg['optimizer']['weight_decay'],
        betas=cfg['optimizer']['betas'] if 'betas' in cfg['optimizer'] else [0.9, 0.999],
    )

    T_max = cfg['optimizer']['iterations'] // cfg['dataset']['batch_size']
    warmup_phase = T_max // 10

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=T_max,
        cycle_mult=1.0,
        max_lr=cfg['optimizer']['lr']['max_lr'],
        min_lr=cfg['optimizer']['lr']['min_lr'],
        warmup_steps=warmup_phase,
        gamma=1.0,
    )

    ckp_manager = main_utils.CheckpointManager(model_dir, rank=args.rank)

    # Optionally resume from a checkpoint
    if cfg['resume']:
        if ckp_manager.checkpoint_exists(last=True):
            start_epoch = ckp_manager.restore(
                restore_last=True, model=model, optimizer=optimizer, train_criterion=train_criterion
            )
            scheduler.step(start_epoch)
            logger.info("Checkpoint loaded: '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch))
        else:
            logger.info("No checkpoint found at '{}'".format(ckp_manager.last_checkpoint_fn()))

    cudnn.benchmark = True

    ############################ TRAIN #########################################
    test_freq = cfg['test_freq'] if 'test_freq' in cfg else 1

    # switch to train mode
    model.train()

    lr_meter = metrics_utils.AverageMeter('LR', ':.2e')
    batch_time = metrics_utils.AverageMeter('Time', ':.2f', window_size=100)
    loss_meter = metrics_utils.AverageMeter('Loss', ':.3f')

    print('length of dataset: ', len(dataset))
    progress = utils.logger.ProgressMeter_Iteration(
        len(dataset),
        [lr_meter, batch_time, loss_meter],
        phase='train',
        logger=logger,
        tb_writter=tb_writter,
    )

    end = time.time()

    memorybank = FVMemoryBank(
        memory_size=len(dataset),
        embedding_dim=cfg['model']['args']['proj_dim'],
        momentum=cfg['loss']['memory_momentum'],
        device=0,
    )
    repr(memorybank)

    iteration = 0

    for batch, sample in enumerate(train_loader):

        # placeholder for clustering result
        cluster_result = {'inst2cluster': [], 'centroids': [], 'density': []}
        for num_cluster in cfg['clustering']['num_cluster']:
            cluster_result['inst2cluster'].append(torch.zeros(len(dataset), dtype=torch.long).cuda())
            cluster_result['centroids'].append(torch.zeros((int(num_cluster), cfg['model']['args']['proj_dim'])).cuda())
            cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())

        frame_cluster_result = None
        audio_cluster_result = None
        if (
            batch % (cfg['clustering']['clustering_iter_freq'] // cfg['dataset']['batch_size']) == 0
            and iteration >= cfg['clustering']['warmup_iter']
        ):
            model.eval()

            features_voice = memorybank.view1_mem
            features_face = memorybank.view2_mem
            print('Memory: ', memorybank.view1_mem.shape, memorybank.view2_mem.shape)

            frame_cluster_result = run_kmeans(
                features_face,
                cfg['clustering']['num_cluster'],
                cfg['clustering']['Niter'],
                temperature=0.2,
                verbose=True,
            )
            audio_cluster_result = run_kmeans(
                features_voice,
                cfg['clustering']['num_cluster'],
                cfg['clustering']['Niter'],
                temperature=0.2,
                verbose=True,
            )
            # save the clustering result
            # torch.save(cluster_result, os.path.join(args.exp_dir, 'clusters_%d'%iteration))

        model.train()

        # Prepare batch
        audio, frame = sample['audio'], sample['frame']
        video_index = sample['video_index']

        bs = audio.size(0)

        audio = audio.cuda(non_blocking=True)  # (bs, 1, 64, nframes)
        frame = frame.cuda(non_blocking=True)  # (bs, 3, 224, 224)
        video_index = video_index.cuda(non_blocking=True)
        # print('input: ', audio.shape, frame.shape)

        # compute audio and video embeddings
        audio_emb, frame_emb = model(audio, frame)
        # print('output: ', audio_emb.shape, frame_emb.shape)

        # compute audio and video memories
        audio_mem, frame_mem = memorybank(audio_emb, frame_emb, video_index)

        # compute loss
        loss = train_criterion(audio_emb, frame_emb, audio_cluster_result, frame_cluster_result, video_index)

        loss_meter.update(loss.item(), audio_emb.size(0))

        # compute gradient and do SGD step during training
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print to terminal and tensorboard
        if batch % cfg['print_freq'] == 0 or iteration == 0 or iteration >= cfg['optimizer']['iterations'] - bs:
            progress.display(iteration)
            # logger.info('LR: {:.2e}'.format(scheduler.get_last_lr()[0]))

        # Valuation and save checkpoint
        if batch % (cfg['test_freq']) == 0 or iteration >= cfg['optimizer']['iterations'] - bs:
            ckp_manager.save(
                iteration, eval_metric=-loss, model=model, optimizer=optimizer, train_criterion=train_criterion
            )

        scheduler.step()
        lr_meter.update(scheduler.get_lr()[0], 1)

        iteration += bs
        if iteration >= cfg['optimizer']['iterations']:
            break


if __name__ == '__main__':
    main()
