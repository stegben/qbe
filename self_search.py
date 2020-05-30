import argparse
import os

from dotenv import load_dotenv

from aqbe.data import LibriSpeechWithAlignment, Data
from aqbe.audio_loader import TorchAudio
from aqbe.encoder import MFCC
from aqbe.index import SimpleRails


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
        os.path.join(audio_directory, 'train-clean-100', '1*/**/*.flac'),
        os.path.join(alignment_directory, 'train-clean-100', '**/*.txt'),
    )
    audio_loader = TorchAudio()
    encoder = MFCC()

    data = Data(audio_loader, audio_provider, encoder)

    index = SimpleRails.build_from_data(
        data,
        n_hough_peaks=100,
        n_nearest_frames=100,
        offset_merge_threshold=10,
    )

    for _ in range(10000):
        feature, idxs = data.sample_range(length=1.5)
        result = index.query(feature)

    import ipdb; ipdb.set_trace()
