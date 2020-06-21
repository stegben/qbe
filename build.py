import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import pickle as pkl
import sys

from tqdm import tqdm

from factories import (
    attach_build_args,
    attach_save_args,
    build_data_from_args,
    build_index,
)

REGISTRY_DIR = os.environ.get('QBE_REGISTRY_DIR', '/tmp')
NOW = datetime.now().strftime('%Y%m%d-%H%M%S')

def build_argparse():
    parser = argparse.ArgumentParser()
    parser = attach_build_args(parser)
    parser = attach_save_args(parser)
    return parser


if __name__ == '__main__':
    args = build_argparse().parse_args()

    print('=========== Create data...')
    data = build_data_from_args(args)

    print('=========== Create index...')
    index = build_index(args, data)

    print('=========== Start building index...')
    for feature, idx in tqdm(data.generate(), desc='Build index...'):
        index.add(feature, idx)

    print('=========== Finish building index, start saving...')
    saved_dir_name = f'{args.save_path_prefix}_{NOW}'
    saved_dir = os.path.join(REGISTRY_DIR, saved_dir_name)
    saved_dir = Path(saved_dir)
    saved_dir.mkdir(parents=True, exist_ok=False)

    with open(str(saved_dir / 'command_line.txt'), 'w') as fw:
        fw.write(' '.join(sys.argv[1:]))

    with open(str(saved_dir / 'args.json'), 'w') as fw:
        json.dump(vars(args), fw)

    with open(str(saved_dir / 'data.pkl'), 'wb') as fw:
        pkl.dump(data, fw)

    index.save(str(saved_dir / 'index'))
