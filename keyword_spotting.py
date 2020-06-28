import argparse
import os
import random

import numpy as np
from tqdm import tqdm

from aqbe.eval import match_ratio
from factories import (
    attach_load_args,
    attach_query_args,
    prepare_data_index_for_query,
    prepare_test_data,
)


def keyword_spotting_argparse():
    parser = argparse.ArgumentParser()
    attach_load_args(parser)
    attach_query_args(parser)
    return parser


if __name__ == '__main__':
    args = keyword_spotting_argparse().parse_args()
    data, index = prepare_data_index_for_query(args)
    test_data = prepare_test_data(data, args)

    results = []
    all_query_word = []
    all_preds = []
    all_score = []
    query_words = test_data.most_frequent_words(200)[100:]
    for _ in tqdm(range(args.n_queries)):

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
