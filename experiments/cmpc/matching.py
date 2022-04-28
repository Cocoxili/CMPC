import sys

sys.path.append("/home/cocoxili/Res/CMPC")

import argparse
from utils.main_utils import *
from utils.logger import *
from utils import metrics_utils

from dataloader_verfication import *
from verification import load_model
import models
from IPython import embed

parser = argparse.ArgumentParser(description='Matching task.')
parser.add_argument('cfg', metavar='CFG', help='config file')
parser.add_argument('--ckp_path', default='', type=str, help='Path of the checkpoint.')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use.')


def matching(cfg, logger):
    """
    1:N matching
    """
    logger.info("=" * 30 + "   Matching   " + "=" * 30)

    seed_everything(cfg['seed'])

    if args.ckp_path != '':
        checkpoint_fn = args.ckp_path
    else:
        checkpoint_fn = cfg['matching']['checkpoint_fn']
        checkpoint_fn = os.path.join(
            cfg['root_path'], cfg['checkpoint_path'], os.path.split(os.getcwd())[-1], checkpoint_fn
        )

    model = load_model(cfg, checkpoint_fn, logger)

    for matching_file in cfg['matching']['v2f_files']:
        matching_file = os.path.join(cfg['root_path'], matching_file)
        matching_set = VoxCeleb1_V2F_Matching(matching_file, cfg)
        matching_on_file(cfg, matching_file, matching_set, 'v2f', model, logger)

    for matching_file in cfg['matching']['f2v_files']:
        matching_file = os.path.join(cfg['root_path'], matching_file)
        matching_set = VoxCeleb1_F2V_Matching(matching_file, cfg)
        matching_on_file(cfg, matching_file, matching_set, 'f2v', model, logger)  # exchange the subnet

    # for matching_file in cfg['matching']['1:N_files']:
    #     matching_file = os.path.join(cfg['root_path'], matching_file)
    #     matching_set = VoxCeleb1_V2F_Matching(matching_file, cfg)
    #     matching_on_file(cfg, matching_file, matching_set, model, logger)


def matching_on_file(cfg, matching_file, matching_set, direction, model, logger):
    N = matching_set.N

    # voice_embednet, face_embednet, projnet = model.voice_subnet, model.face_subnet, model.projnet
    voice_embednet, face_embednet = model.voice_subnet, model.face_subnet

    matching_loader = DataLoader(
        matching_set,
        batch_size=cfg['matching']['batch_size'],
        num_workers=cfg['matching']['num_workers'],
        shuffle=False,
    )

    acc_meter = metrics_utils.AverageMeter('ACC', ':.3f')

    for idx, sample in enumerate(matching_loader):
        probe, candidate = sample['probe'], sample['candidate']
        probe, candidate = probe.cuda(), candidate.cuda()  # (bs, 1, h, w), (bs, N, 3, h, w)
        bs = probe.size(0)
        candidate = candidate.view(bs * candidate.size(1), candidate.size(2), candidate.size(3), candidate.size(4))

        with torch.no_grad():
            if direction == 'v2f':
                probe_emb = voice_embednet(probe)
                candidate_emb = face_embednet(candidate)
                # if cfg['matching']['proj']:
                #     probe_emb = projnet(probe_emb)
                #     candidate_emb = projnet(candidate_emb)

            elif direction == 'f2v':
                probe_emb = face_embednet(probe)
                candidate_emb = voice_embednet(candidate)
                # if cfg['matching']['proj']:
                #     probe_emb = projnet(probe_emb)
                #     candidate_emb = projnet(candidate_emb)

        probe_emb = probe_emb.expand(N, -1, -1).transpose(0, 1)  # (bs, N, 512)
        candidate_emb = candidate_emb.view(bs, -1, candidate_emb.size(1))  # (bs, N, 512)

        sim = nn.CosineSimilarity(dim=2, eps=1e-8)
        scores = sim(probe_emb, candidate_emb)  # (bs, N)

        _, pred = torch.topk(scores, dim=1, k=1, largest=True)
        correct = bs - torch.count_nonzero(pred)
        acc = correct / bs
        acc_meter.update(acc.item(), bs)
        if idx % 10 == 0:
            logger.info("{}/{}: {}".format(idx, len(matching_set) // cfg['matching']['batch_size'], str(acc_meter)))
    logger.info("{}: {}\n".format(matching_file, str(acc_meter)))


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    logger = create_logging('./test_log', filemode='a')
    logger.info(os.path.abspath(__file__))
    logger.info("=" * 30 + "   Config   " + "=" * 30)
    logger.info('\n' + pprint.pformat(cfg))

    matching(cfg, logger)
