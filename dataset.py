#-*- coding : utf-8-*-
# coding:unicode_escape
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


full_keys = ['EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft', 'EyeLookUpLeft', 'EyeSquintLeft', 'EyeWideLeft', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight', 'EyeLookUpRight', 'EyeSquintRight',
        'EyeWideRight', 'JawForward', 'JawRight', 'JawLeft', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthRight', 'MouthLeft', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft',
        'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft',
        'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut', 'HeadYaw', 'HeadPitch',
        'HeadRoll', 'LeftEyeYaw', 'LeftEyePitch', 'LeftEyeRoll', 'RightEyeYaw', 'RightEyePitch', 'RightEyeRoll']

keys = ['JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthLeft', 'MouthRight', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft',
        'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft',
        'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight']


class BlendshapeDataset(Dataset):

    def __init__(self, feature_file, target_file):
        wav_feature = np.load(feature_file)
        val_wav=np.load('/home/pzr/code/aiwin/data/audio2face_data_for_train/train/eval/x_val.npy')
        self.wav_feature = np.vstack((wav_feature, val_wav))
        # self.wav_feature=wav_feature

        # reshape to avoid automatic conversion to doubletensor
        target=np.load(target_file) # / 100.0
        val_bs=np.load('/home/pzr/code/aiwin/data/audio2face_data_for_train/train/eval/y_val_37.npy')
        target=pd.DataFrame(target, columns=full_keys)
        target=target[keys]
        target=target.values

        self.blendshape_target = np.vstack((target, val_bs))
        # self.blendshape_target=target

        self._align()

    def __len__(self):
        return len(self.wav_feature)

    def _align(self):
        """
            align audio feature with blendshape feature
            generally, number of audio feature is less
        """

        n_audioframe, n_videoframe = len(self.wav_feature), len(self.blendshape_target)
        print('Current dataset -- n_videoframe: {}, n_audioframe:{}'.format(n_videoframe, n_audioframe))
        assert n_videoframe - n_audioframe <= 40
        if n_videoframe != n_audioframe:
            start_videoframe = 16
            self.blendshape_target = self.blendshape_target[start_videoframe : start_videoframe+n_audioframe]

    def __getitem__(self, index):
        wav=self.wav_feature[index]
        wav=np.squeeze(wav)
        wav=wav.T
        wav=np.expand_dims(wav, axis=0)
        # print(wav.shape)
        return wav, self.blendshape_target[index]

class AudioDataset(Dataset):
    def __init__(self, feature_file):
        self.wav_feature = np.load(feature_file)
        # reshape to avoid automatic conversion to doubletensor

    def __len__(self):
        return len(self.wav_feature)

    def __getitem__(self, index):
        wav=self.wav_feature[index]
        wav=np.squeeze(wav)
        wav=wav.T
        wav=np.expand_dims(wav, axis=0)
        # print(wav.shape)
        return wav

class Blendshape37Dataset(Dataset):

    def __init__(self, feature_file, target_file):
        self.wav_feature = np.load(feature_file)
        self.blendshape_target = np.load(target_file)
        # reshape to avoid automatic conversion to doubletensor


        self._align()

    def __len__(self):
        return len(self.wav_feature)

    def _align(self):
        """
            align audio feature with blendshape feature
            generally, number of audio feature is less
        """

        n_audioframe, n_videoframe = len(self.wav_feature), len(self.blendshape_target)
        print('Current dataset -- n_videoframe: {}, n_audioframe:{}'.format(n_videoframe, n_audioframe))
        assert n_videoframe - n_audioframe <= 40
        if n_videoframe != n_audioframe:
            start_videoframe = 16
            self.blendshape_target = self.blendshape_target[start_videoframe : start_videoframe+n_audioframe]

    def __getitem__(self, index):
        wav=self.wav_feature[index]
        wav=np.squeeze(wav)
        wav=wav.T
        wav=np.expand_dims(wav, axis=0)
        # print(wav.shape)
        return wav, self.blendshape_target[index]
