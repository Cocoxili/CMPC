import datetime
import sys
import logging
import os

import torch
from torch import distributed as dist


def create_logging(log_dir, filemode):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    now = datetime.datetime.now()
    log_fn = now.strftime("%Y%m%d-%H%M%S") + ".log"

    log_path = os.path.join(log_dir, log_fn)
    logging.basicConfig(
        level=logging.DEBUG,
        # format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%m%d-%H:%M:%S',
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m%d-%H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging


class ProgressMeter(object):
    def __init__(self, num_batches, meters, phase, epoch=None, logger=None, tb_writter=None):
        self.batches_per_epoch = num_batches
        self.batch_fmtstr = self._get_batch_fmtstr(epoch, num_batches)
        self.meters = meters
        self.phase = phase
        self.epoch = epoch
        self.logger = logger
        self.tb_writter = tb_writter

    def display(self, batch):
        step = self.epoch * self.batches_per_epoch + batch
        entries = ['{}'.format(self.batch_fmtstr.format(batch))]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('  '.join(entries))

        if self.tb_writter is not None:
            for meter in self.meters:
                self.tb_writter.add_scalar('{}-batch/{}'.format(self.phase, meter.name), meter.val, step)

    def _get_batch_fmtstr(self, epoch, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        epoch_str = '[{}]'.format(epoch) if epoch is not None else ''
        return epoch_str + '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def synchronize_meters(self, cur_gpu):
        metrics = torch.tensor([m.avg for m in self.progress.meters]).cuda(cur_gpu)
        metrics_gather = [torch.ones_like(metrics) for _ in range(dist.get_world_size())]
        dist.all_gather(metrics_gather, metrics)

        metrics = torch.stack(metrics_gather).mean(0).cpu().numpy()
        for meter, m in zip(self.progress.meters, metrics):
            meter.avg = m


class ProgressMeter_Iteration(object):
    def __init__(self, num_samples, meters, phase, logger=None, tb_writter=None):
        self.num_samples = num_samples
        self.meters = meters
        self.phase = phase
        self.logger = logger
        self.tb_writter = tb_writter

    def display(self, iteration):
        epoch = iteration // self.num_samples
        entries = ['[{}][{}]'.format(epoch, iteration)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('  '.join(entries))

        if self.tb_writter is not None:
            for meter in self.meters:
                self.tb_writter.add_scalar('{}-iteration/{}'.format(self.phase, meter.name), meter.val, iteration)
