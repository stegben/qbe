import os
from pathlib import Path
import pickle as pkl

from dotenv import load_dotenv

from aqbe.audio_loader import TorchAudio
from aqbe.encoder import MFCC
from aqbe.index import SimpleRails
from aqbe.data import LibriSpeechWithAlignment, Data

load_dotenv()


KEY2ENCODER = {
    'MFCC': MFCC,
}

KEY2INDEX = {
    'SimpleRails': SimpleRails,
}

KEY2AUDIO_LOADER = {
    'TorchAudio': TorchAudio,
}


LIBRISPEECH_PATH = os.environ['QBE_LIBRISPEECH_PATH']
LIBRIALIGNED_PATH = os.environ['QBE_LIBRIALIGNED_PATH']
AUDIO_PROVIDERS = {
    'libri-100': LibriSpeechWithAlignment(
        os.path.join(LIBRISPEECH_PATH, 'train-clean-100', '*/*/*.flac'),
        os.path.join(LIBRIALIGNED_PATH, 'train-clean-100', '*/*/*.txt'),
    ),
    'libri-360': LibriSpeechWithAlignment(
        os.path.join(LIBRISPEECH_PATH, 'train-clean-360', '*/*/*.flac'),
        os.path.join(LIBRIALIGNED_PATH, 'train-clean-360', '*/*/*.txt'),
    ),
    'libri-test': LibriSpeechWithAlignment(
        os.path.join(LIBRISPEECH_PATH, 'test-clean', '*/*/*.flac'),
        os.path.join(LIBRIALIGNED_PATH, 'test-clean', '*/*/*.txt'),
    ),

    # for debug
    'sample-libri': LibriSpeechWithAlignment(
        os.path.join(LIBRISPEECH_PATH, 'train-clean-100', '9*/*/*.flac'),
        os.path.join(LIBRIALIGNED_PATH, 'train-clean-100', '9*/*/*.txt'),
    ),
}
REGISTRY_DIR = os.environ.get('QBE_REGISTRY_DIR', '/tmp')


class ClsSelector:

    def __init__(self, key2cls, default_key):
        self._key2cls = key2cls
        self._default_key = default_key
        if self._default_key not in self._key2cls:
            k = self._default_key
            allowed_keys = ', '.join(self._key2cls)
            raise KeyError(f'{k} is not in [{allowed_keys}]')

    def create(self, key, **kwargs):
        key = key or self._default_key
        cls = self._key2cls[key]
        return cls(**kwargs)


def build_data_from_args(args):

    audio_provider = AUDIO_PROVIDERS[args.data]

    audio_loader_factory = ClsSelector(KEY2AUDIO_LOADER, 'TorchAudio')
    audio_loader = audio_loader_factory.create(args.loader)

    encoder_factory = ClsSelector(KEY2ENCODER, 'MFCC')
    encoder_key = args.encoder
    if encoder_key == 'MFCC':
        encoder_kwargs = {key[5:]: value for key, value in vars(args).items() if key[:5] == 'mfcc_'}
    else:
        encoder_kwargs = {}
    encoder = encoder_factory.create(encoder_key, **encoder_kwargs)

    data = Data(audio_loader, audio_provider, encoder)
    return data


def build_index(args, data):
    index_factory = ClsSelector(KEY2INDEX, 'SimpleRails')

    key = args.index
    if key == 'SimpleRails':
        kwargs = {key[3:]: value for key, value in vars(args).items() if key[:3] == 'sr_'}
        kwargs = {
            'dim': data.feature_dims,
            'total_frames': data.n_frames,
            **kwargs,
        }
    else:
        kwargs = {}
    index = index_factory.create(key, **kwargs)

    return index


def prepare_data_index_for_query(args):
    loaded_dir_name = args.load_from
    loaded_dir = os.path.join(REGISTRY_DIR, loaded_dir_name)
    loaded_dir = Path(loaded_dir)

    with open(str(loaded_dir / 'data.pkl'), 'rb') as f:
        data = pkl.load(f)

    index = SimpleRails.load(str(loaded_dir / 'index'))
    kwargs = {key[3:]: value for key, value in vars(args).items() if key[:3] == 'sr_'}
    index.set_query_params(**kwargs)
    return data, index


def prepare_test_data(data, args):
    audio_provider = AUDIO_PROVIDERS[args.test_data]
    return Data(
        data.audio_loader,
        audio_provider,
        data.encoder,
    )


def attach_build_args(parser):
    parser.add_argument(
        '--data',
        type=str,
        choices=AUDIO_PROVIDERS.keys(),
        required=True,
    )

    # Eencoders
    parser.add_argument(
        '--encoder',
        type=str,
        choices=KEY2ENCODER.keys(),
        required=True,
    )
    parser.add_argument(
        '--mfcc_sample_rate',
        type=int,
        default=16000,
    )
    parser.add_argument(
        '--mfcc_n_mfcc',
        type=int,
        default=39,
    )
    parser.add_argument(
        '--mfcc_n_mels',
        type=int,
        default=39,
    )
    parser.add_argument(
        '--mfcc_n_fft',
        type=int,
        default=400,
    )
    parser.add_argument(
        '--mfcc_hop_length',
        type=int,
        default=400,
    )
    parser.add_argument(
        '--mfcc_win_length',
        type=int,
        default=160,
    )
    parser.add_argument(
        '--mfcc_cmvn',
        action='store_true',
        default=False,
    )

    # Audio Loaders
    parser.add_argument(
        '--loader',
        type=str,
        choices=KEY2AUDIO_LOADER.keys(),
        default='TorchAudio',
    )

    # Indexs
    parser.add_argument(
        '--index',
        type=str,
        choices=KEY2INDEX.keys(),
        required=True,
    )
    parser.add_argument(
        '--sr_hnsw_space',
        type=str,
        default='l2',
        choices=['l2', 'cosine'],
    )
    parser.add_argument(
        '--sr_hnsw_ef_construction',
        type=int,
        default=200,
    )
    parser.add_argument(
        '--sr_hnsw_M',
        type=int,
        default=16,
    )
    return parser


def attach_query_args(parser):
    parser.add_argument(
        '--n_queries',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--test_data',
        type=str,
        choices=AUDIO_PROVIDERS.keys(),
    )

    # SimpleRAILS query params
    parser.add_argument(
        '--sr_ef',
        type=int,
        default=200,
    )
    parser.add_argument(
        '--sr_n_nearest_frames',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--sr_n_hough_peaks',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--sr_offset_merge_threshold',
        type=int,
        default=10,
    )
    return parser


def attach_save_args(parser):
    parser.add_argument(
        '--save_path_prefix',
        type=str,
        default='',
        required=True,
    )
    return parser


def attach_load_args(parser):
    parser.add_argument(
        '--load_from',
        type=str,
        required=True,
    )
    return parser
