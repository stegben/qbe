from collections import namedtuple
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import torch
import torchaudio as ta


LIBRI_SPEECH_PATH = '/mnt/data/datasets/LibriSpeech/LibriSpeech'
LIBRI_TTS_PATH = '/mnt/data/datasets/LibriTTS/LibriTTS'
LIBRI_ALIGNED_PATH = '/mnt/data/datasets/LibriAligned/LibriSpeech'

Alignment = namedtuple('Alignment', ['words', 'secs'])


if __name__ == "__main__":

    # import audio data
    # split data into query and candidates
    libri_speech_folder = Path(LIBRI_SPEECH_PATH)
    all_voice_path = list(libri_speech_folder.glob('train-clean-100/*/*/*.flac'))
    all_voice_path = all_voice_path[:1000]
    file2voice = {}
    file2mfcc = {}

    mfcc = ta.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=39,
        melkwargs=dict(
            win_length=4000,
            n_fft=4000,
            hop_length=1600,
            n_mels=39,
        ),
    )
    for voice_path in tqdm(all_voice_path):
        voice, sr = ta.load(voice_path)
        assert sr == 16000
        file_name = str(voice_path).split('/')[-1]
        file_name = file_name[:-5]  # remove '.flac' to match alignments
        file2voice[file_name] = voice
        file2mfcc[file_name] = mfcc(voice)


    libri_aligned_folder = Path(LIBRI_ALIGNED_PATH)
    all_aligned_texts_path = list(libri_aligned_folder.glob('train-clean-100/*/*/*.txt'))

    file2alignment = {}
    for aligned_texts_path in all_aligned_texts_path:
        with aligned_texts_path.open('r') as f:
            for line in f:
                raw = line.rstrip().split()
                voice_file_name = raw[0]
                if voice_file_name not in file2voice:
                    continue
                words = raw[1].replace('"', '').split(',')
                secs = raw[2].replace('"', '').split(',')
                secs = [float(s) for s in secs]
                assert len(secs) == len(words)
                file2alignment[voice_file_name] = Alignment(words=words, secs=secs)

    import ipdb; ipdb.set_trace()
