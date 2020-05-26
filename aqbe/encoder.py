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

    def __init__(self):
        self.mfcc = ta.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=39,
            melkwargs=dict(
                win_length=400,
                n_fft=400,
                hop_length=160,
                n_mels=39,
            ),
        )

    def encode(self, audio):
        audio_tensor = torch.from_numpy(audio)
        feature = self.mfcc(audio_tensor)
        feature.transpose_(0, 1)
        return feature.numpy()

    @property
    def dim(self):
        return 39

    def to_seconds(self, n_frames: int):
        return (n_frames * 160 + 240) / 16000

    def to_frames(self, secs: Second) -> int:
        return int(secs * 16000 // 160)
