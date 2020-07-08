""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys

# Compute confidence intervals
import statsmodels.stats.api as sms


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

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
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    exact_match = total = 0
    f1_scores = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1_scores.append(
                    metric_max_over_ground_truths(f1_score, prediction, ground_truths))
    
    # Exact-match is binary, so use binomial CI's
    exact_match_mean = 100.0 * exact_match / total
    lower, upper = sms.proportion_confint(exact_match, total, alpha=0.05, method="beta")
    exact_match_ci = (100.0 * lower, 100.0 * upper)

    # F1 scores are continuous in [0, 1]. We could use fancy bounded random
    # variable CI, but for now, we settle for the normal approximation.
    f1_mean = 100.0 * sum(f1_scores) / total
    lower, upper = sms.DescrStatsW(f1_scores).tconfint_mean()
    f1_ci = (100.0 * lower, 100.0 * upper)

    return {'exact_match': exact_match_mean,
            'exact_match_ci': exact_match_ci,
            'f1': f1_mean,
            'f1_ci': f1_ci}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ')
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))