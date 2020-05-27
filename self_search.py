import os

from dotenv import load_dotenv

from aqbe.data import LibriSpeechWithAlignment, Data
from aqbe.audio_loader import TorchAudio
from aqbe.encoder import MFCC
from aqbe.index import SimpleRails


if __name__ == '__main__':
    load_dotenv()

    audio_directory = os.environ['QBE_LIBRISPEECH_PATH']
    alignment_directory = os.environ['QBE_LIBRIALIGNED_PATH']

    audio_provider = LibriSpeechWithAlignment(audio_directory, alignment_directory)
    audio_loader = TorchAudio()
    encoder = MFCC()

    data = Data(audio_loader, audio_provider, encoder)

    index = SimpleRails.build_from_data(
        data,
        n_hough_peaks=100,
        n_nearest_frames=100,
        offset_merge_threshold=10,
    )

    import ipdb; ipdb.set_trace()
