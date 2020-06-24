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
""" Finetuning the library models for question-answering on SQuAD (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import logging
import math
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from .huggingface.pytorch_transformers import (
    BertConfig,
    XLMConfig,
    XLMTokenizer,
    XLNetConfig,
)
from .mrqa_modeling_bert import (
    MRQABertTokenizer,
    BertForMRQAQuestionAnswering,
)
from .mrqa_modeling_xlm import XLMForMRQAQuestionAnswering
from .mrqa_modeling_xlnet import (
    XLNetForMRQAQuestionAnswering,
    MRQAXLNetTokenizer,
)
from .utils.utils_mrqa import (
    RawResult,
    RawResultExtended,
    get_bert_text_predictions,
    get_xlnet_text_predictions,
    read_mrqa_datasets,
    split_examples_into_segments,
)

# from utils_squad import (read_squad_examples, convert_examples_to_features,
#                          RawResult, write_predictions,
#                          RawResultExtended, write_predictions_extended)

# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)
# from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, XLNetConfig, XLMConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForMRQAQuestionAnswering, MRQABertTokenizer),
    "xlnet": (XLNetConfig, XLNetForMRQAQuestionAnswering, MRQAXLNetTokenizer),
    "xlm": (XLMConfig, XLMForMRQAQuestionAnswering, XLMTokenizer),
}


def predict(args, assets, data_paths, include_labels=False):
    results = defaultdict(dict)

    model, tokenizer, device, n_gpu = (
        assets["model"],
        assets["tokenizer"],
        assets["device"],
        assets["n_gpu"],
    )

    eval_dataset = read_mrqa_datasets(data_paths, args.eval_sample_size)
    eval_batch_size = args.per_gpu_eval_batch_size * max(1, n_gpu)

    all_predictions = {}
    eval_examples = eval_dataset["examples"]

    train_example_segments, features_per_segment, examples_per_segment = load_and_cache_features(
        args,
        eval_examples,
        tokenizer,
        eval_batch_size,
        is_training=False,
        include_labels=include_labels,
        serial=not args.multiprocessing["featurization"],
        workers=args.multiprocessing["eval_featurization_workers"],
    )

    eval_features = train_example_segments[0]

    # Note that DistributedSampler samples randomly
    eval_inputs = vectorize(eval_features, include_labels=include_labels)
    eval_sampler = (
        SequentialSampler(eval_inputs)
        if args.local_rank == -1
        else DistributedSampler(eval_inputs)
    )
    eval_dataloader = DataLoader(eval_inputs, sampler=eval_sampler, batch_size=eval_batch_size)

    all_results = []
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[1],
                "attention_mask": batch[2],
                "token_type_ids": None
                if args.model_type == "xlm"
                else batch[3],  # XLM don't use segment_ids
            }
            example_indices = batch[0]
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.model_type in ["xlnet", "xlm"]:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(
                    unique_id=unique_id,
                    start_top_log_probs=to_list(outputs[0][i]),
                    start_top_index=to_list(outputs[1][i]),
                    end_top_log_probs=to_list(outputs[2][i]),
                    end_top_index=to_list(outputs[3][i]),
                    cls_logits=to_list(outputs[4][i]),
                )
            else:
                result = RawResult(
                    unique_id=unique_id,
                    start_logits=to_list(outputs[0][i]),
                    end_logits=to_list(outputs[1][i]),
                )
            all_results.append(result)

    output_prediction_file, output_nbest_file, output_null_log_odds_file, predict_file = (
        None,
        None,
        None,
        None,
    )

    if args.model_type in ["xlnet", "xlm"]:
        # XLNet uses a more complex post-processing procedure
        if hasattr(model, "module"):  # Model may be wrapped in DataParallel
            start_n_top = model.module.config.start_n_top
            end_n_top = model.module.config.end_n_top
        else:
            start_n_top = model.config.start_n_top
            end_n_top = model.config.end_n_top
        dataset_predictions = get_xlnet_text_predictions(
            eval_examples,
            eval_features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            predict_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose,
        )
    else:
        dataset_predictions = get_bert_text_predictions(
            eval_examples,
            eval_features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
        )

    prev_len = len(all_predictions)
    all_predictions.update(dataset_predictions)
    assert prev_len + len(dataset_predictions) == len(all_predictions)

    results["predictions"] = all_predictions
    logger.info(all_predictions)
    return results


def setup_model(args):
    device, n_gpu = setup_cuda(args.local_rank, args.seed, args.fp16)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name is not None else args.model_name_or_path,
        config_overrides=getattr(args, "model_config_overrides", {}),
    )
    mrqa_special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[TLE]", "[DOC]", "[PAR]"]
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name is not None else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        do_basic_tokenize=True,
        never_split=mrqa_special_tokens,
        additional_special_tokens=mrqa_special_tokens,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Distributed and parrallel training
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Training/evaluation parameters %s", args)
    return {"model": model, "tokenizer": tokenizer, "device": device, "n_gpu": n_gpu}


def setup_cuda(local_rank: int, seed: int, fp16: bool) -> (str, int):
    """Sets up cuda, and the random seed."""
    # Setup CUDA, GPU & distributed training
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        local_rank,
        device,
        n_gpu,
        bool(local_rank != -1),
        fp16,
    )

    set_seed(seed, n_gpu)
    return device, n_gpu


def load_and_cache_features(
    args,
    all_examples,
    tokenizer,
    batch_size,
    is_training=False,
    include_labels=False,
    serial=False,
    workers=None,
):
    num_example_split = args.num_example_split if is_training else 1

    random.shuffle(all_examples)  # We shuffle all examples together here before segmenting
    examples_per_segment = math.ceil(len(all_examples) / num_example_split)
    train_example_segments, features_per_segment = split_examples_into_segments(
        all_examples=all_examples,
        num_example_split=num_example_split,
        examples_per_segment=examples_per_segment,
        batch_size=batch_size,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        output_dir=args.output_dir if args.save_cache_features else None,
        is_training=include_labels,
        verbose=args.verbose,
        serial=serial,
        workers=workers,
    )

    return train_example_segments, features_per_segment, examples_per_segment


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def vectorize(features, include_labels: bool = True) -> TensorDataset:
    """Wraps each feature in a Tensor and packages then in a TensorDataset."""
    all_example_ids = torch.arange(len(features), dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    input_tensors = [
        all_example_ids,
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_cls_index,
        all_p_mask,
    ]
    if include_labels:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        input_tensors += [all_start_positions, all_end_positions]
    return TensorDataset(*input_tensors)
