import abc

import torch
import torchaudio as ta

from .types import VoiceType, AudioFeatureType


class EncoderBase(abc.ABC):

    @abc.abstractmethod
    def encode(self, voice: VoiceType) -> AudioFeatureType:
        pass

    @property
    @abc.abstractmethod
    def dim(self):
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

    def encode(self, voice):
        voice_tensor = torch.from_numpy(voice)
        voice_tensor.transpose_(0, 1)
        return voice_tensor.numpy(0)

    @property
    def dim(self):
        return 39
