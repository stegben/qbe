import abc
from collections import namedtuple, defaultdict
from pathlib import Path

import numpy as np

from .types import AlignmentType
from .utils import RangeLookup


Alignment = namedtuple('Alignment', ['word', 'start_sec', 'end_sec'])
Position = namedtuple('Position', ['key', 'start', 'end'])


class LibriSpeechWithAlignment:

    def __init__(self, audio_directory, alignment_directory):
        libri_speech_folder = Path(audio_directory)
        libri_aligned_folder = Path(alignment_directory)

        self.all_voice_path = list(libri_speech_folder.glob('train-clean-100/*/*/*.flac')) \
                            + list(libri_speech_folder.glob('train-clean-360/*/*/*.flac'))
        self.all_aligned_texts_path = list(libri_aligned_folder.glob('train-clean-100/*/*/*.txt')) \
                                    + list(libri_aligned_folder.glob('train-clean-360/*/*/*.txt'))

        self.key2path = {}
        self.key2alignments = defaultdict(list)

        for aligned_texts_path in self.all_aligned_texts_path:
            with aligned_texts_path.open('r') as f:
                for line in f:
                    raw = line.rstrip().split()
                    key = raw[0]
                    words = raw[1].replace('"', '').split(',')
                    secs = raw[2].replace('"', '').split(',')
                    secs = [float(s) for s in secs]
                    assert len(secs) == len(words)

                    start_sec = 0
                    for word, end_sec in zip(words, secs):
                        self.key2alignments[key].append(Alignment(
                            word=word,
                            start_sec=start_sec,
                            end_sec=end_sec,
                        ))
                        start_sec = end_sec

        for voice_path in self.all_voice_path:
            key = self.gen_key(voice_path)
            self.key2path[key] = voice_path

        self._keys = sorted(list(set().union(self.key2path, self.key2alignments)))

    @property
    def keys(self):
        return self._keys

    def gen_key(self, path):
        return path.stem

    def get_audio_path(self, key):
        return self.key2path[key]

    def get_alignments(self, key):
        return self.key2alignments[key]


class Data:
    """Provided encoded features of audio files, and provide inverse lookup
    """
    def __init__(self, audio_loader, audio_provider, encoder):
        self.audio_loader = audio_loader
        self.audio_provider = audio_provider
        self.encoder = encoder

        self.idx2key = RangeLookup()
        self.idx2word = RangeLookup()
        self.key2feature = {}
        self.word2positions = defaultdict(list)

        self.encoded_audio_features = []
        self._n_frames = 0
        for key in self.audio_provider.keys:
            audio_path = self.audio_provider.get_audio_path(key)
            audio = self.audio_loader.extract_audio(audio_path, start_sec=0, end_sec=None)
            feature = self.encoder.encode(audio)
            n_frame = feature.shape[0]
            self._n_frames += n_frame
            self.idx2key.add(self._n_frames, key)
            self.key2feature[key] = feature

            alignments = self.audio_provider.get_alignments(key)

            for word, start_sec, end_sec in alignments:
                start_frames = self.encoder.to_frames(start_sec)
                end_frames = self.encoder.to_frames(end_sec)
                self.word2positions[word].append(Position(
                    key=key,
                    start=start_frames,
                    end=end_frames,
                ))
                start_sec = end_sec

    def generate(self, *_, **__):
        start_idx = 0
        for key in self.audio_provider.keys:
            feature = self.key2feature[key]
            n_frames = feature.shape[0]
            idxs = np.arange(n_frames) + start_idx
            yield feature, idxs
            start_idx += n_frames

    @property
    def n_frames(self):
        return self._n_frames

    @property
    def feature_dims(self):
        return self.encoder.dim

    def extract(self, key, start_sec, end_sec=None):
        path = self.audio_provider.get_path[key]
        audio = self.audio_loader.extract_audio(path, start_sec=start_sec, end_sec=end_sec)
        return self.encoder.encode(audio)

    # def reverse_lookup(self, start_frame_idx, end_frame_idx):
    #     key, n_frames = self.idx2key[start_frame_idx, end_frame_idx]
    #     seconds = self.encoder.to_seconds(n_frames)
    #     return key, seconds
