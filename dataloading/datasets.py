# Imports
import os
import numpy as np
import torch
from torchaudio import load
from torch.utils.data import Dataset
from src.source.preprocess import make_features
import decord as de
import cv2


class EmbeddingsDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.audio = data['audio']
        self.video = data['video']

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        audio = torch.from_numpy(self.audio[idx])
        video = torch.from_numpy(self.video[idx])

        return audio, video

