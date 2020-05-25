import abc
from typing import Tuple, Any

import torchaudio as ta

from .types import LoadedVoiceType


class AudioLoaderBase(abc.ABC):

    @abc.abstractmethod
    def _load(self, path: str, sampling_rate: int = None) -> LoadedVoiceType:
        pass

    def extract_voice(
            self,
            path: str,
            start_sec: int = 0,
            end_sec: int = None,
            sampling_rate: int = None,
        ):
        voice, sr = self._load(path, sampling_rate)
        start = int(sr * start_sec)
        if end_sec is None:
            return voice[start:]
        else:
            end = int(sr * end_sec)
            return voice[start:end]


class TorchAudio(AudioLoaderBase):

    def _load(self, path: str, sampling_rate: int = None):
        tensor, raw_sampling_rate = ta.load(path)
        if sampling_rate is not None:
            # TODO: resample
            pass
        return tensor.numpy()[0], raw_sampling_rate


class Librosa(AudioLoaderBase):
    pass


class PyAudio(AudioLoaderBase):
    pass
