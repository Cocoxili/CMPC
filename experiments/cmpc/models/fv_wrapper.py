import torch
import torch.nn as nn
import torch.nn.functional as F
from .fv_resnet import *


class Head(nn.Module):
    def __init__(self, last_dim, proj_dims):
        super(Head, self).__init__()

        projection = [nn.Dropout(0.5), nn.Linear(last_dim, proj_dims)]

        self.projection = nn.Sequential(*projection)
        self.out_dim = proj_dims

    def forward(self, x):
        return self.projection(x)


class FV_Wrapper(nn.Module):
    def __init__(self, pretrain, last_dim, proj_dim, proj, test_only=False):
        super(FV_Wrapper, self).__init__()
        self.test_only = test_only
        self.proj = proj
        self.voice_subnet = resnet34(input_channel=1, out_dim=last_dim, pretrain=pretrain)
        self.face_subnet = resnet34(input_channel=3, out_dim=last_dim, pretrain=pretrain)

        # self.projnet = Head(last_dim=last_dim, proj_dims=proj_dim)

        # self.voice_subnet_proj = nn.Sequential(self.voice_subnet, self.projnet)
        # self.face_subnet_proj = nn.Sequential(self.face_subnet, self.projnet)
        # self.test_only = test_only

    def forward(self, audio, frame):
        audio_emb = self.voice_subnet(audio)
        # if self.proj:
        #     audio_emb = self.projnet(audio_emb)
        frame_emb = self.face_subnet(frame)
        # if self.proj:
        #     frame_emb = self.projnet(frame_emb)
        return audio_emb, frame_emb
