from typing import List

import numpy as np


def overlap_ratio(x_start, x_end, y_start, y_end):
    x_len = x_end - x_start
    y_len = y_end - y_start

    overlap_start = max(x_start, y_start)
    overlap_end = min(x_end, y_end)
    return max(overlap_end - overlap_start, 0.) / max(x_len, y_len)


def match_ratio(
        truth: str,
        preds: List[List[str]],
        correct_threshold: float = 0.3,
    ):
    scores = []
    for pred in preds:
        match_ratio = sum(truth == _p for _p in pred) / len(pred)
        score = match_ratio > correct_threshold
        scores.append(int(score))
    return np.mean(scores)


# TODO: standard metrics
def OTWV():
    pass


def MAP():
    pass


def precision_at_k():
    pass
