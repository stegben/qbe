import bisect
import os
from collections import namedtuple, Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from hnswlib import Index
from tqdm import tqdm
import torch
import torchaudio as ta

load_dotenv(verbose=True)


LIBRI_SPEECH_PATH = os.environ.get('QBE_LIBRISPEECH_PATH')
LIBRI_ALIGNED_PATH = os.environ.get('QBE_LIBRIALIGNED_PATH')

Alignment = namedtuple('Alignment', ['words', 'secs'])


class RangeLookup:

    def __init__(self):
        self.positions = []
        self.labels = []

    def add(self, position, label):
        if self.positions:
            if position < self.positions[-1]:
                raise ValueError(f'Adding new element should be incremetal. Got {position}: {label}')
        self.positions.append(position)
        self.labels.append(label)

    def __getitem__(self, position):
        idx = bisect.bisect_left(self.positions, position)
        return self.labels[idx]


class HoughAccumulations:

    def __init__(self):
        self.counts = Counter()
        self.key2labels = defaultdict(list)

    def add(self, slope, offset, label):
        key = self.hash(slope, offset)
        self.counts.update([key])
        self.key2labels[key].append(label)

    def peaks(self, k):
        candidates = self.counts.most_common(k)
        result = []
        for key, count in candidates:
            result.append((count, self.key2labels[key]))
        return result

    def hash(self, x, y):
        return (int(x), int(y))


if __name__ == "__main__":

    # import audio data
    # split data into query and candidates
    n_files = 1000
    libri_speech_folder = Path(LIBRI_SPEECH_PATH)
    all_voice_path = list(libri_speech_folder.glob('train-clean-100/*/*/*.flac'))
    all_voice_path = all_voice_path[:n_files]
    file2voice = {}
    file2mfcc = {}

    mfcc = ta.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=39,
        melkwargs=dict(
            win_length=400,
            n_fft=400,
            hop_length=160,
            n_mels=39,
        ),
    )
    total_frames = 0
    for voice_path in tqdm(all_voice_path):
        voice, sr = ta.load(voice_path)
        assert sr == 16000
        file_name = str(voice_path).split('/')[-1]
        file_name = file_name[:-5]  # remove '.flac' to match alignments
        # file2voice[file_name] = voice
        frame_features = mfcc(voice)[0]
        frame_features.transpose_(0, 1)
        file2mfcc[file_name] = frame_features
        n_frames = frame_features.shape[0]
        total_frames += n_frames
    index2file = RangeLookup()
    index = Index(space='l2', dim=39)
    index.init_index(max_elements=total_frames, ef_construction=200, M=16)
    start = 0
    for file_name, feature in tqdm(file2mfcc.items()):
        n_frames = feature.shape[0]
        index.add_items(feature.numpy(), start + np.arange(n_frames))
        start += n_frames
        index2file.add(start, file_name)
    index.set_ef(200)

    sample_file = all_voice_path[n_files // 2]
    voice, _= ta.load(sample_file)
    start = 20045  # intentionally not divided by 160
    query_voice = voice[:, start:(start + 16000*3)]
    query_features = mfcc(query_voice)[0]
    query_features.transpose_(0, 1)
    query_features = query_features.numpy()
    m_frames = query_features.shape[0]
    knn_points = []
    for m_idx in range(m_frames):
        frame = query_features[m_idx, :]
        nearest_frames, _distances = index.knn_query(frame, k=100)
        knn_points.append(nearest_frames[0])

    # Hough transform
    accumulations = HoughAccumulations()
    for m_idx, n_idxs in enumerate(knn_points):
        # slope constraint
        slope_candidates = [1]
        for slope in slope_candidates:
            for n_idx in list(n_idxs):
                offset = slope * -m_idx + n_idx
                accumulations.add(slope, offset, n_idx)
    import ipdb; ipdb.set_trace()

    # libri_aligned_folder = Path(LIBRI_ALIGNED_PATH)
    # all_aligned_texts_path = list(libri_aligned_folder.glob('train-clean-100/*/*/*.txt'))

    # file2alignment = {}
    # for aligned_texts_path in all_aligned_texts_path:
    #     with aligned_texts_path.open('r') as f:
    #         for line in f:
    #             raw = line.rstrip().split()
    #             voice_file_name = raw[0]
    #             if voice_file_name not in file2voice:
    #                 continue
    #             words = raw[1].replace('"', '').split(',')
    #             secs = raw[2].replace('"', '').split(',')
    #             secs = [float(s) for s in secs]
    #             assert len(secs) == len(words)
    #             file2alignment[voice_file_name] = Alignment(words=words, secs=secs)

    # import ipdb; ipdb.set_trace()
