import os
import glob
import librosa
import pickle
import argparse
import numpy as np
from easydict import EasyDict
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Audio data transformation.')
parser.add_argument(
    '--wav_dir',
    default='/home/cocoxili/Dataset/VoxCeleb1/wav',
    type=str,
    help='Source path of the wav format.',
)
parser.add_argument(
    '--logmel_dir',
    default='/home/cocoxili/work/VoxCeleb1/logmel',
    type=str,
    help='Destination path of the logmel file.',
)


def save_data(filename, data):
    """Save variable into a pickle file

    Parameters
    ----------
    filename: str
        Path to file

    data: list or dict
        Data to be saved.

    Returns
    -------
    nothing

    """
    pickle.dump(data, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(data, open(filename, 'w'))


def load_data(filename):
    """Load data from pickle file

    Parameters
    ----------
    filename: str
        Path to file

    Returns
    -------
    data: list or dict
        Loaded file.

    """

    return pickle.load(open(filename, "rb"), encoding='latin1')


def wav_to_logmel(cfg):
    files_path = glob.glob(f"{cfg.wav_dir}/**/*.wav", recursive=True)
    # files_path = glob.glob(f"{cfg.dataset_path}/[!'_background_noise_']**/*.wav", recursive=True)
    print("Total number of data: ", len(files_path))  # 153516
    # files_path = files_path[:5]
    print(files_path[:5])
    pool = Pool(10)
    pool.map(tsfm_logmel, iter(files_path))
    # tqdm(pool.imap(tsfm_logmel, iter(files_path)), total=len(files_path))


def tsfm_logmel(src_path):
    fn = os.path.splitext(src_path.split('wav/', 1)[1])[0] + '.pkl'
    dst_path = os.path.join(cfg.logmel_dir, fn)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    print(dst_path)

    n_fft = int(cfg['frame_width'] / 1000 * cfg['sampling_rate'])
    hop_length = int(cfg['frame_shift'] / 1000 * cfg['sampling_rate'])

    data, sr = librosa.load(src_path, cfg.sampling_rate)
    # data, _ = librosa.effects.trim(data)
    melspec = librosa.feature.melspectrogram(
        data, cfg['sampling_rate'], n_fft=n_fft, hop_length=hop_length, n_mels=cfg['n_mels']
    )
    logmel = librosa.power_to_db(melspec)
    logmel = logmel.astype(np.float32)
    # print(logmel)
    # print(logmel.shape)
    assert logmel.shape[0] > 10

    save_data(dst_path, logmel)


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = EasyDict()
    cfg.sampling_rate = 16000
    cfg.frame_width = 100
    cfg.frame_shift = 10
    cfg.n_mels = 64
    cfg.wav_dir = args.wav_dir
    cfg.logmel_dir = args.logmel_dir

    wav_to_logmel(cfg)
