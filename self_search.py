import argparse
import os

from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

from aqbe.data import LibriSpeechWithAlignment, Data
from aqbe.audio_loader import TorchAudio
from aqbe.encoder import MFCC
from aqbe.index import SimpleRails
from aqbe.eval import overlap_ratio


def self_search_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data',
        type=str,
        default=''
    )
    parser.add_argument
    return parser


if __name__ == '__main__':
    load_dotenv()

    audio_directory = os.environ['QBE_LIBRISPEECH_PATH']
    alignment_directory = os.environ['QBE_LIBRIALIGNED_PATH']

    audio_provider = LibriSpeechWithAlignment(
        os.path.join(audio_directory, 'train-clean-100', '*/*/*.flac'),
        os.path.join(alignment_directory, 'train-clean-100', '*/*/*.txt'),
    )
    audio_loader = TorchAudio()
    encoder = MFCC()

    data = Data(audio_loader, audio_provider, encoder)

    index = SimpleRails(
        dim=data.feature_dims,
        total_frames=data.n_frames,
        n_hough_peaks=100,
        n_nearest_frames=100,
        offset_merge_threshold=10,
    )
    for feature, idx in tqdm(data.generate(), desc='Build index...'):
        index.add(feature, idx)

    results = []
    labels = []
    correct = []
    query_length = 1.5
    for _ in tqdm(range(100)):
        try:
            key, start_sec, feature = data.sample_range(length=query_length)
        except:
            continue
        try:
            score, start_frame_idx, end_frame_idx = index.query(feature)[0]
        except:
            continue
        labels.append((key, start_sec))
        results.append((score, start_frame_idx, end_frame_idx))

        query_result = data.reverse_lookup(start_frame_idx, end_frame_idx)
        if len(query_result) > 2:
            correct.append(0.)
        elif key != query_result[0][0]:
            correct.append(0.)
        else:
            query_start_sec = query_result[0][1]
            query_end_sec = query_result[1][1]

            score = overlap_ratio(
                start_sec, start_sec + query_length,
                query_start_sec, query_end_sec,
            )
            if score > 0.8:
                correct.append(1)
            else:
                correct.append(0)
    print(f'score: {np.mean(correct)}')
    import ipdb; ipdb.set_trace()
