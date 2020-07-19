import json
import logging
import os
import collections
from functools import partial
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from ...file_utils import is_tf_available, is_torch_available
from ...tokenization_bert import whitespace_tokenize
from .utils import DataProcessor


ProcessExampleArgs = collections.namedtuple(
    "ProcessExampleArgs",
    [
        "tokenizer",
        "max_seq_length",
        "doc_stride",
        "max_query_length",
        "is_training",
        "verbose",
        "cls_token_at_end",
        "cls_token",
        "sep_token",
        "pad_token",
        "sequence_a_segment_id",
        "sequence_b_segment_id",
        "cls_token_segment_id",
        "pad_token_segment_id",
        "mask_padding_with_zero",
    ],
)

# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart"}


if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def mrqa_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def process_example(example_index, example, args):
    query_tokens = args.tokenizer.mrqa_tokenize_tokens(example.question_tokens)

    if len(query_tokens) > args.max_query_length:
        query_tokens = query_tokens[0 : args.max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        # sub_tokens = tokenizer.tokenize(token)
        sub_tokens = args.tokenizer.mrqa_tokenize_tokens([token])
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if args.is_training and example.is_impossible:
        tok_start_position = -1
        tok_end_position = -1
    if args.is_training and not example.is_impossible:
        # TODO: Use more detected answers?
        start_position = example.detected_answers[0]["token_spans"][0][0]
        end_position = example.detected_answers[0]["token_spans"][0][1]

        tok_start_position = orig_to_tok_index[start_position]
        if end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens,
            tok_start_position,
            tok_end_position,
            args.tokenizer,
            example.detected_answers[0]["text"],  # example.orig_answer_texts,
        )

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = args.max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"]
    )
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, args.doc_stride)

    features = []
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = []

        # CLS token at the beginning
        if not args.cls_token_at_end:
            tokens.append(args.cls_token)
            segment_ids.append(args.cls_token_segment_id)
            p_mask.append(0)
            cls_index = 0

        # Query
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(args.sequence_a_segment_id)
            p_mask.append(1)

        # SEP token
        tokens.append(args.sep_token)
        segment_ids.append(args.sequence_a_segment_id)
        p_mask.append(1)

        # Paragraph
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(args.sequence_b_segment_id)
            p_mask.append(0)
        paragraph_len = doc_span.length

        # SEP token
        tokens.append(args.sep_token)
        segment_ids.append(args.sequence_b_segment_id)
        p_mask.append(1)

        # CLS token at the end
        if args.cls_token_at_end:
            tokens.append(args.cls_token)
            segment_ids.append(args.cls_token_segment_id)
            p_mask.append(0)
            cls_index = len(tokens) - 1  # Index of classification token

        input_ids = args.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if args.mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < args.max_seq_length:
            input_ids.append(args.pad_token)
            input_mask.append(0 if args.mask_padding_with_zero else 1)
            segment_ids.append(args.pad_token_segment_id)
            p_mask.append(1)

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length

        span_is_impossible = example.is_impossible
        start_position = None
        end_position = None
        if args.is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
                span_is_impossible = True
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        if args.is_training and span_is_impossible:
            start_position = cls_index
            end_position = cls_index

        if example_index < 5 and args.verbose:
            if args.is_training and not span_is_impossible:
                answer_text = " ".join(tokens[start_position : (end_position + 1)])

        mrqa_metadata = {
            "qas_id": example.qas_id,
        }

        features.append(
            InputFeatures(
                unique_id=0,
                metadata=mrqa_metadata,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                cls_index=cls_index,
                p_mask=p_mask,
                paragraph_len=paragraph_len,
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
            )
        )

    return features


def mrqa_convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        cls_token_at_end=False,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token=0,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        cls_token_segment_id=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        verbose=True,
        serial=False,
        workers=None,
):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    args = ProcessExampleArgs(
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        verbose,
        cls_token_at_end,
        cls_token,
        sep_token,
        pad_token,
        sequence_a_segment_id,
        sequence_b_segment_id,
        cls_token_segment_id,
        pad_token_segment_id,
        mask_padding_with_zero,
    )
    if serial:
        example_features = [
            process_example(i, example, args) for (i, example) in enumerate(examples)
        ]
    else:
        with Pool(workers) as p:
            example_features = p.starmap(
                process_example,
                [(i, example, args) for (i, example) in enumerate(examples)],
                chunksize=64,
            )

    flattened_features = []
    for features in example_features:
        for feat in features:
            feat.unique_id = unique_id
            unique_id += 1
            flattened_features.append(feat)

    return flattened_features



class MRQAProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    Overriden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples::

            >>> import tensorflow_datasets as tfds
            >>> dataset = tfds.load("squad")

            >>> training_examples = get_examples_from_dataset(dataset, evaluate=False)
            >>> evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )

                    examples.append(example)
        return examples


class SquadV1Processor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class SquadV2Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"


class MRQAExample(object):
    """A single training/test example for the MRQA dataset."""

    def __init__(
            self,
            qas_id,
            question_tokens,
            doc_tokens,
            orig_answer_texts=None,
            context=None,
            context_tokens=None,
            # start_position=None,
            # end_position=None,
            is_impossible=None,
            detected_answers=None,
            # answer_texts=None,
    ):
        self.qas_id = qas_id
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.orig_answer_texts = orig_answer_texts
        # self.start_position = start_position
        # self.end_position = end_position
        self.is_impossible = is_impossible
        self.detected_answers = detected_answers
        self.context = context
        self.context_tokens = context_tokens

    def __str__(self):
        return self.__repr__()


class MRQAFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            unique_id,
            metadata,
            example_index,
            doc_span_index,
            tokens,
            token_to_orig_map,
            token_is_max_context,
            input_ids,
            input_mask,
            segment_ids,
            cls_index,
            p_mask,
            paragraph_len,
            start_position=None,
            end_position=None,
            is_impossible=None,
    ):
        self.unique_id = unique_id
        self.metadata = metadata
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits
