import sys

sys.path.append("/home/cocoxili/Res/CMPC")

import argparse
from sklearn import metrics
import matplotlib.pyplot as plt

from utils.main_utils import *
from utils.logger import *
from dataloader_verfication import *
from utils.torch_utils import *
import models


parser = argparse.ArgumentParser(description='Verification Evaluation.')
parser.add_argument('cfg', metavar='CFG', help='config file')
parser.add_argument('--ckp_path', default='', type=str, help='Path of the checkpoint.')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use.')


def get_AUC(labels, scores):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    return metrics.auc(fpr, tpr)


def get_EER(labels, scores):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer


def plot_auc_curve(fpr, tpr):
    auc = metrics.auc(fpr, tpr)
    plt.title('AUC Curve')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def get_score_and_label(test_file, cfg, voice_embednet, face_embednet):
    scores = []
    labels = []

    proj = cfg['verification']['proj']
    testset = VoxCeleb1_Testpair(test_file, cfg)
    testloader = DataLoader(
        testset, batch_size=cfg['verification']['batch_size'], num_workers=cfg['verification']['num_workers']
    )
    print(f'{len(testset)} test pairs in {os.path.split(test_file)[-1]}')
    for idx, sample in enumerate(testloader):
        voice, face, label = sample['audio'], sample['frame'], sample['label']
        # [bs, voice_aug_times, 1, h, w], [bs, face_aug_times, 3, 224, 224], [bs]
        bs = voice.size(0)
        voice, face = voice.cuda(), face.cuda()
        voice = voice.view(bs * voice.size(1), voice.size(2), voice.size(3), voice.size(4))
        face = face.view(bs * face.size(1), face.size(2), face.size(3), face.size(4))
        # [bs*times, 1, h, w], [bs*times, 3, 224, 224]

        with torch.no_grad():
            voice_emb = voice_embednet(voice)
            face_emb = face_embednet(face)

        voice_emb = voice_emb.view(bs, -1, voice_emb.size(1))
        face_emb = face_emb.view(bs, -1, face_emb.size(1))
        # [bs, times, dim], [bs, times, dim]
        voice_emb = torch.mean(voice_emb, dim=1)
        face_emb = torch.mean(face_emb, dim=1)

        batch_score = batchwise_cosine_distance(voice_emb, face_emb)  # (bs)

        scores.extend(batch_score.cpu().tolist())
        labels.extend(label.tolist())

    return scores, labels


def load_model(cfg, checkpoint_fn, logger):
    model = models.__dict__[cfg['model']['arch']](**cfg['model']['args'])
    ckp = torch.load(checkpoint_fn, map_location='cpu')
    # model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
    model.load_state_dict(ckp['model'])
    # logger.info('Load from {} at iteration #{}'.format(checkpoint_fn, ckp['epoch']))
    model = model.cuda()
    model.eval()

    return model


def verify(cfg, logger):
    logger.info("=" * 30 + "   Verification   " + "=" * 30)

    seed_everything(cfg['seed'])

    if args.ckp_path != '':
        checkpoint_fn = args.ckp_path
    else:
        checkpoint_fn = cfg['verification']['checkpoint_fn']
        checkpoint_fn = os.path.join(
            cfg['root_path'], cfg['checkpoint_path'], os.path.split(os.getcwd())[-1], checkpoint_fn
        )

    model = load_model(cfg, checkpoint_fn, logger)
    voice_embednet, face_embednet = model.voice_subnet, model.face_subnet

    for test_file in cfg['verification']['test_files']:
        test_fn = os.path.split(test_file)[-1]
        scores, labels = get_score_and_label(test_file, cfg, voice_embednet, face_embednet)
        auc = get_AUC(labels, scores)
        eer = get_EER(labels, scores)
        logger.info("{}: {:.3f}, {:.3f}".format(test_fn, auc, eer))


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    logger = create_logging('./test_log', filemode='a')
    logger.info(os.path.abspath(__file__))
    logger.info("=" * 30 + "   Config   " + "=" * 30)
    logger.info('\n' + pprint.pformat(cfg))

    verify(cfg, logger)
