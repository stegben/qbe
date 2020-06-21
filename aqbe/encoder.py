import abc

import torch
import torchaudio as ta

from .types import AudioType, AudioFeatureType, Second


class EncoderBase(abc.ABC):

    @abc.abstractmethod
    def encode(self, audio: AudioType) -> AudioFeatureType:
        pass

    @property
    @abc.abstractmethod
    def dim(self):
        pass

    @abc.abstractmethod
    def to_seconds(self, n_frames: int) -> Second:
        pass

    @abc.abstractmethod
    def to_frames(self, secs: Second) -> int:
        pass


class MFCC(EncoderBase):

    def __init__(
            self,
            sample_rate=16000,
            n_mfcc=39,
            win_length=400,
            hop_length=160,
            n_fft=400,
            n_mels=39,
            cmvn=False,
        ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.cmvn = cmvn
        self.mfcc = ta.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs=dict(
                win_length=self.win_length,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            ),
        )

    def encode(self, audio):
        audio_tensor = torch.from_numpy(audio)
        feature = self.mfcc(audio_tensor)
        feature.transpose_(0, 1)
        if self.cmvn:
            # TODO: unit-norm each channel
            pass
        return feature.numpy()

    @property
    def dim(self):
        return self.n_mels

    def to_seconds(self, n_frames: int):
        return ((n_frames - 1) * self.hop_length + self.win_length) / self.sample_rate

    def to_frames(self, secs: Second) -> int:
        return int(secs * self.sample_rate // self.hop_length)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['mfcc']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.mfcc = ta.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs=dict(
                win_length=self.win_length,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            ),
        )
