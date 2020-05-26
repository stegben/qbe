import abc
import bisect
import os
from collections import namedtuple, Counter, defaultdict, OrderedDict
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

HOUGH_PEAKS = 100
FRAME_K = 100
OFFSET_MERGE_THRESHOLD = 10


if __name__ == "__main__":

    # # import audio data
    # # split data into query and candidates

    # libri_speech_folder = Path(LIBRI_SPEECH_PATH)
    # libri_aligned_folder = Path(LIBRI_ALIGNED_PATH)

    # # load candidate data
    # n_files = 1000
    # all_voice_path = list(libri_speech_folder.glob('train-clean-100/*/*/*.flac'))
    # all_voice_path = all_voice_path[:n_files]
    # file2mfcc = {}

    # total_frames = 0
    # for voice_path in tqdm(all_voice_path):
    #     voice, sr = ta.load(voice_path)
    #     import ipdb; ipdb.set_trace()
    #     assert sr == 16000
    #     file_name = voice_path.stem
    #     # file2voice[file_name] = voice
    #     frame_features = mfcc(voice)[0]
    #     frame_features.transpose_(0, 1)
    #     file2mfcc[file_name] = frame_features
    #     n_frames = frame_features.shape[0]
    #     total_frames += n_frames

    # all_aligned_texts_path = list(libri_aligned_folder.glob('train-clean-100/*/*/*.txt'))
    # file2alignment = {}
    # word2position = defaultdict(list)
    # for aligned_texts_path in all_aligned_texts_path:
    #     with aligned_texts_path.open('r') as f:
    #         for line in f:
    #             raw = line.rstrip().split()
    #             voice_file_name = raw[0]
    #             if voice_file_name not in file2mfcc:
    #                 continue
    #             words = raw[1].replace('"', '').split(',')
    #             secs = raw[2].replace('"', '').split(',')
    #             secs = [float(s) for s in secs]
    #             assert len(secs) == len(words)
    #             file2alignment[voice_file_name] = Alignment(words=words, secs=secs)
    #             start_sec = 0.
    #             for word, end_sec in zip(words, secs):
    #                 word2position[word].append(Position(file_name=voice_file_name, start_sec=start_sec, end_sec=end_sec))
    #                 start_sec = end_sec
    # word_counts = {k: len(v) for k, v in word2position.items()}

    # sample_file = all_voice_path[n_files // 2]
    # voice, _= ta.load(sample_file)
    # start = 20045  # intentionally not divided by 160
    # query_voice = voice[:, start:(start + 16000*3)]
    # query_features = mfcc(query_voice)[0]
    # query_features.transpose_(0, 1)
    # query_features = query_features.numpy()
    # knn_points, _distances = index.knn_query(query_features, k=FRAME_K)

    # # merge too-similar pairs
    # merged = set()
    # result = []
    # for idx, ((_, offset), count, points) in enumerate(candidates):
    #     if idx in merged:
    #         continue
    #     cur_left = min(points)
    #     cur_right = max(points)
    #     cur_count = count
    #     for idx2 in range(idx + 1, HOUGH_PEAKS):
    #         if idx2 in merged:
    #             continue
    #         (_, offset_2), count_2, points_2 = candidates[idx2]
    #         if (offset - offset_2) < OFFSET_MERGE_THRESHOLD:
    #             cur_count += count_2
    #             cur_left = min(cur_left, min(points_2))
    #             cur_right = max(cur_right, max(points_2))
    #             merged.add(idx2)
    #     result.append((cur_count, cur_left, cur_right))  # score, start_frame, end_frame

    # import ipdb; ipdb.set_trace()
