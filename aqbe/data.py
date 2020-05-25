import abc
from collections import namedtuple
from pathlib import Path

from .types import AlignmentType
from .utils import RangeLookup


Alignment = namedtuple('Alignment', ['words', 'secs'])
Position = namedtuple('Position', ['file_name', 'start_sec', 'end_sec'])


class LibriSpeechWithAlignment:

    def __init__(self, audio_directory, alignment_directory):
        libri_speech_folder = Path(audio_directory)
        libri_aligned_folder = Path(alignment_directory)

        self.all_voice_path = list(libri_speech_folder.glob('train-clean-100/*/*/*.flac'))
        self.all_aligned_texts_path = list(libri_aligned_folder.glob('train-clean-100/*/*/*.txt'))

        self.key2path = {}
        self.key2alignments = {}

        for aligned_texts_path in self.all_aligned_texts_path:
            with aligned_texts_path.open('r') as f:
                for line in f:
                    raw = line.rstrip().split()
                    key = raw[0]
                    words = raw[1].replace('"', '').split(',')
                    secs = raw[2].replace('"', '').split(',')
                    secs = [float(s) for s in secs]
                    assert len(secs) == len(words)
                    self.key2alignments[key] = Alignment(words=words, secs=secs)

        for voice_path in self.all_voice_path:
            key = self.gen_key(voice_path)
            self.key2path[key] = voice_path

        self.keys = set().union(self.key2path, self.key2alignments)

    def gen_key(self, path):
        return path.stem


class VoiceLibrary:
    """Provided encoded features of audio files, and provide inverse lookup
    """
    def __init__(self, audio_loader, audio_provider, encoder):
        self.audio_loader = audio_loader
        self.audio_provider = audio_provider
        self.encoder = encoder

        self.key2feature = OrderedDict()
        self.idx2key = RangeLookup()

        self.load = False

        for word, end_sec in zip(words, secs):
            word2position[word].append(Position(
                file_name=voice_file_name,
                start_sec=start_sec,
                end_sec=end_sec,
            ))
            start_sec = end_sec

    @property
    def n_frames(self):
        pass

    def extract(self, key, start, end):
        path = self.audio_provider.get_path[key]
        audio = self.audio_loader.extract_voice(path, start, end)
        return self.encoder.encode(audio)

    def reverse_lookup(self, frame_id) -> :
        return self.idx2key[frame_idx]
