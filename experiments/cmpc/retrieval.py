import sys

sys.path.append("/home/cocoxili/Res/CMPC")

import argparse
from sklearn import metrics
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from utils.main_utils import *
from utils.logger import *

from dataloader_verfication import *
from utils.torch_utils import pairwise_cosine_distance
from verification import load_model
import models


parser = argparse.ArgumentParser(description='Retrieval task.')
parser.add_argument('cfg', metavar='CFG', help='config file')
parser.add_argument('--ckp_path', default='', type=str, help='Path of the checkpoint.')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use.')


def get_gt_matrix(v_id, f_id):

    m = np.zeros([len(v_id), len(f_id)])
    for i, v in enumerate(v_id):
        for j, f in enumerate(f_id):
            if v == f:
                m[i][j] = 1
    return m


def get_map_score(gt_matrix, score_matrix):
    assert gt_matrix.shape == score_matrix.shape
    ap_list = []
    for i in range(gt_matrix.shape[0]):
        ap = metrics.average_precision_score(gt_matrix[i], score_matrix[i])
        ap_list.append(ap)

    map = np.array(ap_list).mean()
    return map


def plot_matrix(m):
    plt.matshow(m)
    plt.show()


def retrieval(cfg, logger):
    logger.info("=" * 30 + "   Retrieval   " + "=" * 30)

    seed_everything(cfg['seed'])

    if args.ckp_path != '':
        checkpoint_fn = args.ckp_path
    else:
        checkpoint_fn = cfg['retrieval']['checkpoint_fn']
        checkpoint_fn = os.path.join(
            cfg['root_path'], cfg['checkpoint_path'], os.path.split(os.getcwd())[-1], checkpoint_fn
        )

    model = load_model(cfg, checkpoint_fn, logger)
    voice_embednet, face_embednet = model.voice_subnet, model.face_subnet

    wav_file = os.path.join(cfg['root_path'], cfg['retrieval']['wav_files'])
    jpg_file = os.path.join(cfg['root_path'], cfg['retrieval']['jpg_files'])

    voice_set = VoxCeleb1_Voice(wav_file, cfg)
    voice_load = DataLoader(
        voice_set, batch_size=cfg['retrieval']['batch_size'], num_workers=cfg['retrieval']['num_workers'], shuffle=False
    )

    face_set = VoxCeleb1_Face(jpg_file, cfg)
    face_load = DataLoader(
        face_set, batch_size=cfg['retrieval']['batch_size'], num_workers=cfg['retrieval']['num_workers'], shuffle=False
    )

    num_v, num_f = len(voice_set), len(face_set)
    print(f'#Voice gallery: {num_v}, #Face gallery: {num_f}')

    v_emb = []
    v_id = []
    for idx, sample in enumerate(voice_load):
        id, data = sample['id'], sample['data']
        data = data.cuda()  # [bs, 1, w, h]

        with torch.no_grad():
            emb = voice_embednet(data)  # [bs, dim]
        v_emb.append(emb)

        v_id.extend(id)
    v_emb = torch.cat(v_emb)  # [num_v, dim]

    f_emb = []
    f_id = []
    for idx, sample in enumerate(face_load):
        id, data = sample['id'], sample['data']
        data = data.cuda()  # [bs, 3, w, h]

        with torch.no_grad():
            emb = face_embednet(data)  # [bs, dim]
        f_emb.append(emb)

        f_id.extend(id)
    f_emb = torch.cat(f_emb)  # [num_f, dim]

    pcd = pairwise_cosine_distance(v_emb, f_emb)
    score_matrix = pcd.cpu().numpy()  # [num_v, num_f]

    gt_matrix = get_gt_matrix(v_id, f_id)

    v2f_map = get_map_score(gt_matrix, score_matrix)
    f2v_map = get_map_score(gt_matrix.T, score_matrix.T)

    random_matrix = np.random.rand(num_v, num_f)
    v2f_r_map = get_map_score(gt_matrix, random_matrix)
    f2v_r_map = get_map_score(gt_matrix.T, random_matrix.T)

    logger.info("chance: v2f_map: {:.4f}, f2v_map: {:.4f}".format(v2f_r_map, f2v_r_map))
    logger.info("v2f_map: {:.4f}, f2v_map: {:.4f}".format(v2f_map, f2v_map))


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    logger = create_logging('./test_log', filemode='a')
    logger.info(os.path.abspath(__file__))
    logger.info("=" * 30 + "   Config   " + "=" * 30)
    logger.info('\n' + pprint.pformat(cfg))

    retrieval(cfg, logger)
