import os
import glob
import numpy as np
import pandas as pd

import random
import librosa
import math
import yaml
import pickle
from pprint import pprint
from itertools import cycle

from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from IPython import embed

pd.options.mode.chained_assignment = None


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


def concat_and_cut_wave(files, cfg):
    """
    Concat n audio files and then cut out a clip from it.
    :param files: audio files list
    :return: an audio clip.
    """
    wav_list = []
    for f in files:
        wav, _ = librosa.load(f, sr=cfg['sampling_rate'])
        wav_list.append(wav)

    concat_wav = np.concatenate(wav_list)

    duration = random.randint(cfg['audio_duration'][0], cfg['audio_duration'][1])
    # Maximum audio length
    max_audio = duration * cfg['sampling_rate']
    audiosize = concat_wav.shape[0]

    if audiosize <= max_audio:
        shortage = math.floor((max_audio - audiosize + 1) / 2)
        concat_wav = np.pad(concat_wav, (shortage, shortage), 'constant', constant_values=0)
        audiosize = concat_wav.shape[0]

    startframe = np.int64(random.random() * (audiosize - max_audio))
    cut_wav = concat_wav[int(startframe) : int(startframe) + max_audio]
    return cut_wav


def get_logmel_from_wav(data, cfg):
    n_fft = int(cfg['frame_width'] / 1000 * cfg['sampling_rate'])
    hop_length = int(cfg['frame_shift'] / 1000 * cfg['sampling_rate'])

    melspec = librosa.feature.melspectrogram(
        data, cfg['sampling_rate'], n_fft=n_fft, hop_length=hop_length, n_mels=cfg['n_mels']
    )
    logmel = librosa.power_to_db(melspec)
    return logmel


def load_face(file, cfg):
    face = Image.open(file).convert('RGB').resize([cfg['face_size'][0], cfg['face_size'][1]])
    face = np.transpose(np.array(face), (2, 0, 1))
    face = ((face - 127.5) / 127.5).astype('float32')
    return face


def load_voice(files, cfg):

    file = random.sample(files, k=1)[0]
    voice_data = load_data(file)
    if cfg['fbank_nframes'] > voice_data.shape[1]:
        shortage = math.floor((cfg['fbank_nframes'] - voice_data.shape[1] + 1) / 2)
        voice_data = np.pad(voice_data, ((0, 0), (shortage, shortage)), 'constant', constant_values=0)
    assert cfg['fbank_nframes'] <= voice_data.shape[1]
    pt = np.random.randint(voice_data.shape[1] - cfg['fbank_nframes'] + 1)
    voice_data = voice_data[:, pt : pt + cfg['fbank_nframes']]
    return voice_data


def select_k_frames(file_list, cfg):
    """
    Randomly select k frames from file_list.
    If k > len(file_list), make sure every element in file_list is selected.
    """
    k = cfg['num_frames']
    if k <= len(file_list):
        k_f_files = random.sample(file_list, k=k)
    else:
        k_f_files = random.sample(file_list, k=len(file_list))
        k_f_files.extend(random.choices(file_list, k=(k - len(file_list))))

    face_list = []
    for f in k_f_files:
        face = load_face(f, cfg)
        face_list.append(face)
    faces = np.stack(face_list)
    return faces


class VoxCeleb1Base(Dataset):
    def __init__(self, cfg, split='Train', shuffle=False):
        self.cfg = cfg
        self.split = split
        self.load_spec = self.cfg['load_spec']
        self.meta_df = self._get_meta_df(cfg)
        # print(self.meta_df)
        self.df = self._collect_df(self.meta_df)
        # print(self.meta_df)
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        # print(self.df)
        self.num_instances = self.df.shape[0]

    def _get_meta_df(self, cfg):
        sd = pd.read_csv(cfg['split_file'], sep=' ', names=['VGGFace1 ID', 'split', 'tmp']).drop('tmp', axis=1)
        md = pd.read_csv(cfg['meta_file'], sep='\t')
        df = pd.merge(sd, md, on='VGGFace1 ID')
        if self.split == 'Train':
            split_df = df[df['split'].isin(cfg['train_split'])]
        elif self.split == 'Test':
            split_df = df[df['split'].isin(cfg['test_split'])]
        pd.set_option('mode.chained_assignment', None)
        return split_df

    def _collect_df(self, meta_df):
        """
        add video_id to df
        """
        meta_df['video_id'] = ''
        for idx, row in meta_df.iterrows():
            id = row['VoxCeleb1 ID']
            if self.load_spec:
                p1 = os.path.join(self.cfg['spec_dir'], id)
            else:
                p1 = os.path.join(self.cfg['wav_dir'], id)
            video_id1 = [os.path.split(path)[-1] for path in glob.glob(f"{p1}/*")]

            name = row['VGGFace1 ID']
            p2 = os.path.join(self.cfg['face_dir'], name)
            video_id2 = [os.path.split(path)[-1] for path in glob.glob(f"{p2}/**/*")]

            assert video_id1 == video_id2
            meta_df['video_id'][idx] = video_id1
            # break
        df = meta_df.explode('video_id')
        df = df.reset_index(drop=True)
        return df

    def id_to_name(self, id):
        return self.meta_df.loc[self.meta_df['VoxCeleb1 ID'] == id]['VGGFace1 ID'].values[0]

    def name_to_id(self, name):
        return self.meta_df.loc[self.meta_df['VGGFace1 ID'] == name]['VoxCeleb1 ID'].values[0]

    def id_to_index(self, id):
        return self.meta_df.loc[self.meta_df['VoxCeleb1 ID'] == id].index.values[0]

    def __repr__(self):
        num_id = len(set(self.df['VoxCeleb1 ID'].values))
        num_name = len(set(self.df['VGGFace1 ID'].values))
        assert num_id == num_name
        num_video = len(set(self.df['video_id'].values))
        num_subdir = len(self.df)
        desc = f"VoxCeleb1: \nIdentity number: {num_id}, Video number: {num_video}, Subdir number: {num_subdir}"
        return desc


class VoxCeleb1(VoxCeleb1Base):
    """
    face:(bs, 3, n_f, H, W)或者(bs, 3, H, W), Voice: (bs, 1, H, W)
    """

    def __init__(self, cfg, split='Train', shuffle=False):
        super().__init__(cfg, split, shuffle)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        ID = self.df['VoxCeleb1 ID'][index]
        spec_path = os.path.join(self.cfg['spec_dir'], self.df['VoxCeleb1 ID'][index], self.df['video_id'][index])
        v_files = glob.glob(f"{spec_path}/*.pkl")  # voice files
        spec = load_voice(v_files, self.cfg)
        audio = spec[np.newaxis, ...]

        # Get Positive frame:
        face_path = os.path.join(self.cfg['face_dir'], self.df['VGGFace1 ID'][index])
        f_files = glob.glob(f"{face_path}/1.6/{self.df['video_id'][index]}/*.jpg")  # face files
        # pprint(f_files)
        # print(len(f_files))
        frames_pos = select_k_frames(f_files, self.cfg)  # (k, 3, H, W)
        frames_pos = np.transpose(frames_pos, (1, 0, 2, 3))  # (3, k, H, W)
        frames_pos = np.squeeze(frames_pos)

        sample = {'audio': audio, 'frame': frames_pos, 'ID': ID}

        sample['video_index'] = int(index)

        sample['person_name'] = self.df['VGGFace1 ID'][index]

        return sample


class VoxCeleb1_SingleModal(VoxCeleb1Base):
    """
    Face dataset or Voice dataset
    """

    def __init__(self, cfg, modal, split='Train', shuffle=False):
        super().__init__(cfg, split=split, shuffle=shuffle)
        self.modal = modal
        self.samples = self.traversal_sample()

        # only used for supervised learning
        split_id = set(self.df['VoxCeleb1 ID'].values)
        split_id = list(split_id)
        split_id.sort()
        self.ID_to_idx = {cls_name: i for i, cls_name in enumerate(split_id)}
        # print(self.ID_to_idx)

    def traversal_sample(self):
        samples = []

        for idx, row in self.df.iterrows():
            if self.modal == 'face':
                walk_dir = os.path.join(self.cfg['face_dir'], row['VGGFace1 ID'], '1.6', row['video_id'])
                # print(walk_dir)
            elif self.modal == 'voice':
                walk_dir = os.path.join(self.cfg['spec_dir'], row['VoxCeleb1 ID'], row['video_id'])

            for root, _, fnames in sorted(os.walk(walk_dir, followlinks=True)):
                # print(root, dir, fnames)
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    samples.append(path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = {}
        if self.modal == 'face':
            path = self.samples[index]
            sep_path = path.split(os.sep)
            name = sep_path[-4]
            ID = self.name_to_id(name)
            video_id = sep_path[-2]
            frame = load_face(path, self.cfg)
            sample['data'] = frame

        elif self.modal == 'voice':
            path = self.samples[index]
            sep_path = path.split(os.sep)
            ID = sep_path[-3]
            video_id = sep_path[-2]
            spec = load_voice([path], self.cfg)
            audio = spec[np.newaxis, ...]
            sample['data'] = audio

        video_index = self.df.loc[(self.df['video_id'] == video_id) & (self.df['VoxCeleb1 ID'] == ID)].index[0]
        # print(ID, video_id, video_index)

        sample['video_index'] = video_index
        sample['ID'] = ID
        sample['ID_to_idx'] = self.ID_to_idx[ID]

        return sample


class CycledRandomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        n = len(self.data_source)
        while True:
            yield np.random.randint(0, n)
            # yield random.randint(0, n - 1)

    def __len__(self):
        return len(self.data_source)


class CycledSequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        n = len(self.data_source)
        iter(cycle(range(n)))
        # yield random.randint(0, n - 1)

    def __len__(self):
        return len(self.data_source)


if __name__ == "__main__":
    cfg_file = 'CONFIG.yaml'
    cfg = yaml.safe_load(open(cfg_file))

    dataset = VoxCeleb1_SingleModal(cfg['dataset'], modal='face')
    print(dataset)
    print(len(dataset))

    sampler = CycledRandomSampler(dataset)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        # batch_size=cfg['dataset']['batch_size'],
        batch_size=8,
        shuffle=False,
        num_workers=1,
    )

    for batch, sample in enumerate(loader):
        # audio, frame, ID = sample['audio'], sample['frame'], sample['ID']

        print(sample['audio'].shape, sample['frame'].shape)
        print(sample['person_name'])
        print(sample['ID'])
        print(sample['video_index'])

        if batch == 2:
            break
