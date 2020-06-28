import argparse
import os
from pathlib import Path
import pickle as pkl

import numpy as np
from tqdm import tqdm

from aqbe.eval import overlap_ratio

from factories import attach_load_args, attach_query_args, prepare_data_index_for_query


def self_search_argparse():
    parser = argparse.ArgumentParser()
    attach_query_args(parser)
    attach_load_args(parser)
    return parser


if __name__ == '__main__':
    args = self_search_argparse().parse_args()
    data, index = prepare_data_index_for_query(args)

    results = []
    labels = []
    correct = []
    query_length = 1.5
    for _ in tqdm(range(args.n_queries)):
        key, start_sec, feature = data.sample_range(length=query_length)
        score, start_frame_idx, end_frame_idx = index.query(feature)[0]
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
