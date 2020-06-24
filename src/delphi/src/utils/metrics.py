#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#

from multiprocessing import Pool
import re
import string
from collections import Counter
from typing import Dict, List

import numpy as np


def map_em_value(phrase, answers):
    em_value = metric_max_over_ground_truths(exact_match_score, phrase, answers)
    return float(em_value)


def map_f1_value(phrase, answers):
    f1_value = metric_max_over_ground_truths(f1_score, phrase, answers)
    return float(f1_value)


def get_text_metrics(
    phrases: List[str], answer_texts: List[List[str]], serial=False, workers=None
) -> Dict[str, List[float]]:
    """Compute metrics from the predicted and answer texts."""
    if serial:
        f1_scores = [map_f1_value(phrases[i], answer_texts[i]) for i in range(len(phrases))]
        em_scores = [map_em_value(phrases[i], answer_texts[i]) for i in range(len(phrases))]
    else:
        with Pool(workers) as p:
            f1_scores = p.starmap(
                map_f1_value,
                [(phrases[i], answer_texts[i]) for i in range(len(phrases))],
                chunksize=64,
            )
            em_scores = p.starmap(
                map_em_value,
                [(phrases[i], answer_texts[i]) for i in range(len(phrases))],
                chunksize=64,
            )

    return {"f1": f1_scores, "em": em_scores}


def get_span_metrics(
    predictions: np.ndarray, start_labels: List[int], end_labels: List[int]
) -> Dict[str, List[float]]:
    """Compute span metrics from the predicted indices and gold indices."""
    starts = predictions[:, 0] == start_labels
    ends = predictions[:, 1] == end_labels
    start_acc = starts * 1.0
    end_acc = ends * 1.0
    span_acc = (starts & ends) * 1.0
    return {"start_acc": start_acc, "end_acc": end_acc, "span_acc": span_acc}


####################################################################################
########## OFFICIAL SQUAD EVAL --- DO NOT CHANGE ANYTHING BELOW!!
####################################################################################


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
