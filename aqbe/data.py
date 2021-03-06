import abc
from collections import namedtuple, defaultdict
from glob import glob
from pathlib import Path
import random

import numpy as np
from tqdm import tqdm

from .types import AlignmentType
from .utils import RangeLookup


Alignment = namedtuple('Alignment', ['word', 'start_sec', 'end_sec'])
Position = namedtuple('Position', ['key', 'start', 'end'])


class LibriSpeechWithAlignment:

    def __init__(self, audio_file_pattern, alignment_file_pattern):

        self.all_voice_path = list(glob(audio_file_pattern))
        self.all_aligned_texts_path = list(glob(alignment_file_pattern))

        self.key2path = {}
        self.key2alignments = defaultdict(list)

    def init(self):
        for aligned_texts_path in self.all_aligned_texts_path:
            with open(aligned_texts_path, 'r') as f:
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

        self._keys = sorted(list(set(self.key2path) & set(self.key2alignments)))

    @property
    def keys(self):
        return self._keys

    def gen_key(self, path):
        return Path(path).stem

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
        self.audio_provider.init()
        self.encoder = encoder

        self.idx2key = RangeLookup()
        self.idx2word = RangeLookup()
        self.key2feature = {}
        self.word2positions = defaultdict(list)

        self._n_frames = 0
        for key in tqdm(self.audio_provider.keys, desc='read audio data'):
            audio_path = self.audio_provider.get_audio_path(key)
            audio = self.audio_loader.extract_audio(audio_path, start_sec=0, end_sec=None)

            alignments = self.audio_provider.get_alignments(key)
            for word, start_sec, end_sec in alignments:
                end_frames = self.encoder.to_frames(end_sec)
                if word != '':
                    self.word2positions[word].append(Position(
                        key=key,
                        start=start_sec,
                        end=end_sec,
                    ))
                self.idx2word.add(self._n_frames + end_frames, word)
                start_sec = end_sec

            feature = self.encoder.encode(audio)
            n_frame = feature.shape[0]
            self._n_frames += n_frame
            self.idx2key.add(self._n_frames, key)
            self.key2feature[key] = feature

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
        path = self.audio_provider.get_audio_path(key)
        audio = self.audio_loader.extract_audio(path, start_sec=start_sec, end_sec=end_sec)
        return self.encoder.encode(audio)

    def reverse_lookup(self, start_frame_idx, end_frame_idx):
        points = self.idx2key[start_frame_idx:end_frame_idx]
        results = []
        for key, n_frames in points:
            seconds = self.encoder.to_seconds(n_frames)
            results.append((key, seconds))
        return results

    def sample_range(self, length: float):
        key = random.choice(self.audio_provider.keys)
        feature = self.key2feature[key]
        n_frames = feature.shape[0]
        original_length = self.encoder.to_seconds(n_frames)
        if original_length < length:
            return feature

        start_sec = random.uniform(0., original_length - length)
        end_sec = start_sec + length
        return key, start_sec, self.extract(key, start_sec=start_sec, end_sec=end_sec)

    def most_frequent_words(self, k):
        return sorted(
            self.word2positions,
            key=lambda word: len(self.word2positions[word]),
            reverse=True,
        )[:k]

    def sample_by_word(self, word):
        key, start, end = random.choice(self.word2positions[word])
        return self.extract(key, start, end)

    def ground_truths(self, label):
        return self.word2positions[label]

    def labels(self, start_frame, end_frame):
        raw = self.idx2word[start_frame:end_frame]
        result = []
        for label, _ in raw:
            if result:
                if label == result[-1]:
                    continue
            result.append(label)

        return result
