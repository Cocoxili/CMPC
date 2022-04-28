import os
import shutil
import torch
import numpy as np
import torch.distributed as dist
import datetime
import pprint
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torch.nn as nn
import random
from utils.logger import *


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True


def initialize_distributed_backend(args, ngpus_per_node):
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )

    if args.rank == -1:
        args.rank = 0
    return args


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(os.environ["SLURM_TASKS_PER_NODE"][0])
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)


def prep_environment(args, cfg):
    from torch.utils.tensorboard import SummaryWriter

    # Prepare loggers (must be configured after initialize_distributed_backend())
    exp_name = os.path.split(os.getcwd())[-1]
    model_dir = '{}{}/{}'.format(cfg['root_path'], cfg['model']['model_dir'], exp_name)
    if args.rank == 0:
        prep_output_folder(model_dir, False)
    logger = create_logging(model_dir, filemode='a')
    logger.info(os.path.abspath(__file__))
    logger.info('\n' + pprint.pformat(cfg))

    logger.info('\n' + pprint.pformat(args.__dict__))

    tb_writter = None
    if cfg['log2tb'] and args.rank == 0:
        tb_dir = '{}/tensorboard'.format(model_dir)
        os.system('mkdir -p {}'.format(tb_dir))
        tb_writter = SummaryWriter(tb_dir)

    return logger, tb_writter, model_dir


class CheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.rank = rank
        self.best_metric = -9999.0

    def save(self, epoch, filename=None, eval_metric=-9999.0, **kwargs):
        if self.rank != 0:
            return

        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True

        state = {'epoch': epoch}
        for k in kwargs:
            state[k] = kwargs[k].state_dict()

        if filename is None:
            save_checkpoint(state=state, is_best=is_best, model_dir=self.checkpoint_dir)
        else:
            save_checkpoint(state=state, is_best=is_best, filename='{}/{}'.format(self.checkpoint_dir, filename))

    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def checkpoint_fn(self, last=False, best=False):
        assert best or last
        assert not (last and best)
        if last:
            return self.last_checkpoint_fn()
        if best:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def restore(self, fn=None, restore_last=False, restore_best=False, **kwargs):
        checkpoint_fn = fn if fn is not None else self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        start_epoch = ckp['epoch']
        for k in kwargs:
            if k == 'train_criterion':
                # kwargs[k].load_state_dict(ckp[k], strict=False)
                print('pass')
            else:
                kwargs[k].load_state_dict(ckp[k])
        return start_epoch


def save_checkpoint(state, is_best, model_dir='.', filename=None):
    if filename is None:
        filename = '{}/checkpoint.pth.tar'.format(model_dir)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/model_best.pth.tar'.format(model_dir))


def prep_output_folder(model_dir, evaluate):
    if evaluate:
        assert os.path.isdir(model_dir)
    else:
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
