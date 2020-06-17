import argparse
import os
import random

from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

from aqbe.data import LibriSpeechWithAlignment, Data
from aqbe.audio_loader import TorchAudio
from aqbe.encoder import MFCC
from aqbe.index import SimpleRails
from aqbe.eval import match_ratio


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

    audio_loader = TorchAudio()
    encoder = MFCC()

    audio_provider = LibriSpeechWithAlignment(
        os.path.join(audio_directory, 'train-clean-100', '*/*/*.flac'),
        os.path.join(alignment_directory, 'train-clean-100', '*/*/*.txt'),
    )
    test_audio_provider = LibriSpeechWithAlignment(
        os.path.join(audio_directory, 'test-clean', '*/*/*.flac'),
        os.path.join(alignment_directory, 'test-clean', '*/*/*.txt'),
    )

    data = Data(audio_loader, audio_provider, encoder)
    test_data = Data(audio_loader, test_audio_provider, encoder)

    index = SimpleRails.build_from_data(
        data,
        n_hough_peaks=100,
        n_nearest_frames=100,
        offset_merge_threshold=10,
    )
    results = []
    all_query_word = []
    all_preds = []
    all_score = []
    query_words = test_data.most_frequent_words(200)[100:]
    for _ in tqdm(range(10000)):

        # select a word
        query_word = random.choice(query_words)
        all_query_word.append(query_word)
        # select a sound fragment from test data
        encoded_query = test_data.sample_by_word(query_word)

        # query by the encoded fragment, get the frame start and end
        query_result = index.query(encoded_query)

        # calculate the score
        #### match_ratio
        # get words by the start and end frame
        preds = []
        for _score, start_frame, end_frame in query_result:
            words = data.labels(start_frame, end_frame)
            preds.append(words)
        score = match_ratio(query_word, preds, correct_threshold=1./5.)
        all_score.append(score)
        all_preds.append(preds)

    print(f'score: {np.mean(all_score)}')
    import ipdb; ipdb.set_trace()
