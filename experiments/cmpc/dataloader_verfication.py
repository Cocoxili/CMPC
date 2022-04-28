import os
import glob
import numpy as np
import pandas as pd

import random
import librosa
import math
import yaml
import pprint
import pickle

from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

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


def load_face_aug(file, cfg):
    face = Image.open(file).convert('RGB').resize([cfg['dataset']['face_size'][0], cfg['dataset']['face_size'][1]])
    face = np.transpose(np.array(face), (2, 0, 1))
    face = ((face - 127.5) / 127.5).astype('float32')
    face_ = np.flip(face, axis=2).copy()
    face_aug = np.stack((face, face_))
    return face_aug


def load_face(file, cfg):
    face = Image.open(file).convert('RGB').resize([cfg['face_size'][0], cfg['face_size'][1]])
    face = np.transpose(np.array(face), (2, 0, 1))
    face = ((face - 127.5) / 127.5).astype('float32')
    return face


def load_voice_aug(file, cfg):
    nframes = cfg['dataset']['fbank_nframes']

    voice_data = load_data(file)
    if nframes > voice_data.shape[1]:
        shortage = math.floor((nframes - voice_data.shape[1] + 1) / 2)
        voice_data = np.pad(voice_data, ((0, 0), (shortage, shortage)), 'constant', constant_values=0)
    assert nframes <= voice_data.shape[1]

    voice_aug = []
    for i in range(cfg['verification']['voice_aug_times']):

        pt = np.random.randint(voice_data.shape[1] - nframes + 1)
        voice_clip = voice_data[:, pt : pt + nframes]
        voice_clip = voice_clip[np.newaxis, ...]  # [times, 1, h, w]
        voice_aug.append(voice_clip)
    voice_aug = np.asarray(voice_aug)
    return voice_aug


def load_voice(files, cfg):
    file = random.sample(files, k=1)[0]
    voice_data = load_data(file)
    if cfg['fbank_nframes'] > voice_data.shape[1]:
        shortage = math.floor((cfg['fbank_nframes'] - voice_data.shape[1] + 1) / 2)
        voice_data = np.pad(voice_data, ((0, 0), (shortage, shortage)), 'constant', constant_values=0)
    assert cfg['fbank_nframes'] <= voice_data.shape[1]
    pt = np.random.randint(voice_data.shape[1] - cfg['fbank_nframes'] + 1)
    voice_data = voice_data[:, pt : pt + cfg['fbank_nframes']]
    voice_data = voice_data[np.newaxis, ...]
    return voice_data


class VoxCeleb1_Testpair(Dataset):
    """
    face:(bs, 3, n_f, H, W)或者(bs, 3, H, W), Voice: (bs, 1, H, W)
    """

    def __init__(self, test_file, cfg, shuffle=False):
        self.cfg = cfg
        self.meta_df = self._get_meta_df(cfg['dataset']['meta_file'])
        self.df = self._get_test_df(test_file, self.meta_df)
        # if shuffle:
        #     self.df = self.df.sample(frac=1).reset_index(drop=True)
        # print(self.df)

    def _get_test_df(self, test_file, meta_df):
        test_file = os.path.join(self.cfg['root_path'], test_file)
        df = pd.read_csv(test_file)
        for id, row in df.iterrows():
            # voice-path
            parts = row['voice-path'].split('/')
            voice_id = meta_df.loc[meta_df['VGGFace1 ID'] == parts[0]]['VoxCeleb1 ID'].values[0]
            voice_file = '/'.join([voice_id, parts[1][:11], parts[1][14:-4] + self.cfg['dataset']['voice_ext']])
            voice_file = os.path.join(self.cfg['dataset']['spec_dir'], voice_file)
            df['voice-path'][id] = voice_file

            # face-path
            parts = row['face-path'].split('/')
            face_file = '/'.join(parts[0:1] + ['1.6'] + parts[1:])
            face_file = os.path.join(self.cfg['dataset']['face_dir'], face_file)
            df['face-path'][id] = face_file
        return df

    def _get_meta_df(self, meta_file):
        df = pd.read_csv(meta_file, sep='\t')
        return df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        voice_file = self.df['voice-path'][index]
        voice = load_voice_aug(voice_file, self.cfg)  # [test_aug_times, 1, h, w]

        face_file = self.df['face-path'][index]
        face = load_face_aug(face_file, self.cfg)

        sample = {}
        sample['audio'] = voice
        sample['frame'] = face
        sample['label'] = self.df['label'][index]

        return sample


class VoxCeleb1_V2F_Matching(Dataset):
    """
    1:N voice -> face matching test.
    Voice: (bs, 1, H, W), Face:(bs, N, 3, H, W)
    """

    def __init__(self, matching_file, cfg, shuffle=False):
        self.cfg = cfg
        self.df = pd.read_csv(matching_file, sep=' ', header=None)
        self.N = len(self.df.columns) - 1
        # if shuffle:
        #     self.df = self.df.sample(frac=1).reset_index(drop=True)
        # print(self.df)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        pp = self.df.loc[index, 0].replace('.wav', self.cfg['dataset']['voice_ext'])
        probe_path = os.path.join(self.cfg['dataset']['spec_dir'], pp)
        probe = load_voice([probe_path], self.cfg['dataset'])  # [test_aug_times, 1, h, w]
        candidate = []
        for i in range(self.N):
            cpath = os.path.join(self.cfg['dataset']['face_dir'], self.df.loc[index, i + 1])
            cand = load_face(cpath, self.cfg['dataset'])
            candidate.append(cand)

        candidate = np.array(candidate)

        sample = {}
        sample['probe'] = probe
        sample['candidate'] = candidate

        return sample


class VoxCeleb1_F2V_Matching(Dataset):
    """
    1:N face -> voice matching test.
    Voice: (bs, 1, H, W), Face:(bs, N, 3, H, W)
    """

    def __init__(self, matching_file, cfg, shuffle=False):
        self.cfg = cfg
        self.df = pd.read_csv(matching_file, sep=' ', header=None)
        self.N = len(self.df.columns) - 1
        # if shuffle:
        #     self.df = self.df.sample(frac=1).reset_index(drop=True)
        # print(self.df)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        pp = self.df.loc[index, 0]
        probe_path = os.path.join(self.cfg['dataset']['face_dir'], pp)
        probe = load_face(probe_path, self.cfg['dataset'])
        candidate = []
        for i in range(self.N):
            cp = self.df.loc[index, i + 1].replace('.wav', self.cfg['dataset']['voice_ext'])
            cpath = os.path.join(self.cfg['dataset']['spec_dir'], cp)
            cand = load_voice([cpath], self.cfg['dataset'])
            candidate.append(cand)

        candidate = np.array(candidate)

        sample = {}
        sample['probe'] = probe
        sample['candidate'] = candidate

        return sample


class VoxCeleb1_Voice(Dataset):
    def __init__(self, wav_file, cfg, shuffle=False):
        self.cfg = cfg
        self.df = pd.read_csv(wav_file, sep='\t')
        # if shuffle:
        #     self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        id = self.df.loc[index, ['VoxCeleb1 ID']].values[0]
        pp = self.df.loc[index, ['wav']].values[0].replace('.wav', self.cfg['dataset']['voice_ext'])
        data_path = os.path.join(self.cfg['dataset']['spec_dir'], pp)
        data = load_voice([data_path], self.cfg['dataset'])  # [1, h, w]

        sample = {}
        sample['id'] = id
        sample['data'] = data
        return sample


class VoxCeleb1_Face(Dataset):
    def __init__(self, jpg_file, cfg, shuffle=False):
        self.cfg = cfg
        self.df = pd.read_csv(jpg_file, sep='\t')
        # if shuffle:
        #     self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        id = self.df.loc[index, ['VoxCeleb1 ID']].values[0]
        pp = self.df.loc[index, ['jpg']].values[0]
        data_path = os.path.join(self.cfg['dataset']['face_dir'], pp)
        data = load_face(data_path, self.cfg['dataset'])  # [3, h, w]

        sample = {}
        sample['id'] = id
        sample['data'] = data
        return sample


def cycle(dataloader):
    while True:
        for data, labels in dataloader:
            yield data, labels


class CycledRandomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        n = len(self.data_source)
        while True:
            yield random.randint(0, n - 1)

    def __len__(self):
        return len(self.data_source)
