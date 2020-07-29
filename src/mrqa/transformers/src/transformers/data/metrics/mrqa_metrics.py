# coding=utf-8
#
# File modified on August 9, 2019 by Apple team.
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Load MRQA datasets. """

from __future__ import absolute_import, division, print_function

import json
import collections
import math
import string
import re
from collections import Counter

from transformers.tokenization_bert import BasicTokenizer


RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


def get_bert_text_predictions(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        verbose_logging,
        version_2_with_negative,
        null_score_diff_threshold,
        tokenizer
):
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"],
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(
            prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True
        )

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []

        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]

                # MRQA specific to pull answer from context directly without joining with whitespace
                orig_tokens = example.context_tokens[orig_doc_start : (orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                context_start_index = orig_tokens[0][1]
                context_end_index = orig_tokens[-1][1] + len(orig_tokens[-1][0])

                orig_text = example.context[context_start_index:context_end_index].strip()

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                # orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit
                )
            )
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit, end_logit=null_end_logit
                    )
                )

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = (
                    score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            )
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                # MRQA
                # all_predictions[example.qas_id] = best_non_null_entry.text
                mrqa_final_pred = best_non_null_entry.text
                mrqa_final_pred = mrqa_final_pred.replace("[SEP]", "")
                mrqa_final_pred = mrqa_final_pred.replace("[DOC]", "")
                mrqa_final_pred = mrqa_final_pred.replace("[TLE]", "")
                mrqa_final_pred = mrqa_final_pred.replace("[PAR]", "")
                all_predictions[example.qas_id] = mrqa_final_pred
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


# For XLNet (and XLM which uses the same head)
RawResultExtended = collections.namedtuple(
    "RawResultExtended",
    [
        "unique_id",
        "start_top_log_probs",
        "start_top_index",
        "end_top_log_probs",
        "end_top_index",
        "cls_logits",
    ],
)


def get_xlnet_text_predictions(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        orig_data_file,
        start_n_top,
        end_n_top,
        version_2_with_negative,
        tokenizer,
        verbose_logging,
):
    """ XLNet write prediction logic (more complex than Bert's).
        Write final predictions to the json file and log-odds of null if needed.

        Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_log_prob", "end_log_prob"],
    )

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"]
    )

    # logger.info("Writing predictions to: %s", output_prediction_file)
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = j * end_n_top + i

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob,
                        )
                    )

        prelim_predictions = sorted(
            prelim_predictions, key=lambda x: (x.start_log_prob + x.end_log_prob), reverse=True
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # XLNet un-tokenizer
            # Let's keep it simple for now and see if we need all this later.
            #
            # tok_start_to_orig_index = feature.tok_start_to_orig_index
            # tok_end_to_orig_index = feature.tok_end_to_orig_index
            # start_orig_pos = tok_start_to_orig_index[pred.start_index]
            # end_orig_pos = tok_end_to_orig_index[pred.end_index]
            # paragraph_text = example.paragraph_text
            # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            # Previously used Bert untokenizer
            tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            # orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
            orig_tokens = example.context_tokens[orig_doc_start: (orig_doc_end + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())

            context_start_index = orig_tokens[0][1]
            context_end_index = orig_tokens[-1][1] + len(orig_tokens[-1][0])
            orig_text = example.context[context_start_index:context_end_index].strip()
            # orig_text = " ".join(orig_tokens)

            final_text = get_final_text(
                tok_text, orig_text, tokenizer.do_lower_case, verbose_logging
            )

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob,
                )
            )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="", start_log_prob=-1e6, end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold

        # MRQA
        # all_predictions[example.qas_id] = best_non_null_entry.text
        mrqa_final_pred = best_non_null_entry.text
        mrqa_final_pred = mrqa_final_pred.replace("[SEP]", "")
        mrqa_final_pred = mrqa_final_pred.replace("[DOC]", "")
        mrqa_final_pred = mrqa_final_pred.replace("[TLE]", "")
        mrqa_final_pred = mrqa_final_pred.replace("[PAR]", "")
        all_predictions[example.qas_id] = mrqa_final_pred

        all_nbest_json[example.qas_id] = nbest_json

    # with open(output_prediction_file, "w") as writer:
    #     writer.write(json.dumps(all_predictions, indent=4) + "\n")

    # with open(output_nbest_file, "w") as writer:
    #     writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    # if version_2_with_negative:
    #     with open(output_null_log_odds_file, "w") as writer:
    #         writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    # with open(orig_data_file, "r", encoding='utf-8') as reader:
    #     orig_data = json.load(reader)["data"]

    # qid_to_has_ans = make_qid_to_has_ans(orig_data)
    # has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    # no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    # exact_raw, f1_raw = get_raw_scores(orig_data, all_predictions)
    # out_eval = {}

    # find_all_best_thresh_v2(out_eval, all_predictions, exact_raw, f1_raw, scores_diff_json, qid_to_has_ans)

    return all_predictions  # out_eval

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

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
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths, default=0)


def mrqa_evaluate(examples, predictions):

    f1 = exact_match = total = 0
    #for qid, ground_truths in answers.items():
    for example in examples:
        qid = example.qas_id
        gold_answers = [answer for answer in example.orig_answer_texts if normalize_answer(answer)]
        total += 1
        if qid not in predictions:
            message = 'Unanswered question %s will receive score 0.' % qid
            continue
        prediction = predictions[qid]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, gold_answers)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, gold_answers)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}