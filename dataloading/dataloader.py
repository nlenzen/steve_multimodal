# Imports
import os
import cv2
import csv
import math
import numpy as np
from moviepy.editor import *
import librosa as lb
import torch
import torchaudio
import random
import decord as de
from itertools import chain


class EmbeddingLoader:
    def __init__(self,
                 batch_size,
                 train_data_path,
                 test_data_path,
                 train_val_ratio=(8, 2),
                 overlap=0.75,
                 sample_length=1,
                 n_frames=16,
                 sr=16000,
                 fps=32):
        self.train_video_data = []
        self.train_audio_data = []
        self.test_video_data = []
        self.test_audio_data = []
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.overlap = overlap
        self.sample_length = sample_length
        self.n_frames = n_frames
        self.sr = sr
        self.fps = fps

        if os.path.exists(train_data_path):
            print("Loading training data...")
            data = np.load(train_data_path)
            self.train_video_data = data['video']
            self.train_audio_data = data['audio']

        if os.path.exists(test_data_path):
            print("Loading test data...")
            data = np.load(test_data_path)
            self.test_video_data = data['video']
            self.test_audio_data = data['audio']

    def randomize_sample_order(self):
        p = np.random.permutation(len(self.train_audio_data))
        self.train_video_data = self.train_video_data[p]
        self.train_audio_data = self.train_audio_data[p]

    def len_train_batches(self):
        return math.ceil(len(self.train_video_data) / float(self.batch_size))

    def len_test_batches(self):
        return math.ceil(len(self.test_video_data) / float(self.batch_size))

    def len_test_set(self):
        return len(self.test_video_data)

    def len_train_set(self):
        return len(self.train_video_data)

    # Draws random ramples from the test set. The number of samples is defined by self.batch_size
    def get_random_test_samples(self, size):
        p = np.random.choice(len(self.test_video_data), size)
        video_embeddings = self.test_video_data[p]
        audio_embeddings = self.test_audio_data[p]

        return video_embeddings, audio_embeddings

    def get_random_train_samples(self, size):
        p = np.random.choice(len(self.test_video_data), size)
        video_embeddings = self.train_video_data[p]
        audio_embeddings = self.train_audio_data[p]

        return video_embeddings, audio_embeddings

    def get_batch(self, batch_num):
        start = batch_num * self.batch_size
        end = (batch_num + 1) * self.batch_size
        end = end if end <= len(self.train_video_data) else len(self.train_video_data)

        video_embeddings = self.train_video_data[start: end]
        audio_embeddings = self.train_audio_data[start: end]

        return video_embeddings, audio_embeddings

    def __len__(self):
        return self.len_train_batches()

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.len_train_batches():
            raise StopIteration
        video_embeds, audio_embeds = self.get_batch(self.index)
        self.index += 1
        return video_embeds, audio_embeds


