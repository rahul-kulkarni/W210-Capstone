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

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from .huggingface.pytorch_transformers.modeling_utils import add_start_docstrings
from .huggingface.pytorch_transformers.modeling_xlm import (
    XLMPreTrainedModel,
    XLMModel,
    SQuADHead,
    XLM_START_DOCSTRING,
    XLM_INPUTS_DOCSTRING,
)


@add_start_docstrings(
    """XLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    XLM_START_DOCSTRING,
    XLM_INPUTS_DOCSTRING,
)
class XLMForMRQAQuestionAnswering(XLMPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **is_impossible**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels whether a question has an answer or no answer (SQuAD 2.0)
        **cls_index**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the classification token to use as input for computing plausibility of the answer.
        **p_mask**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        >>> config = XLMConfig.from_pretrained('xlm-mlm-en-2048')
        >>> tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
        >>>
        >>> model = XLMForQuestionAnswering(config)
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> start_positions = torch.tensor([1])
        >>> end_positions = torch.tensor([3])
        >>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        >>> loss, start_scores, end_scores = outputs[:2]

    """

    def __init__(self, config):
        super(XLMForMRQAQuestionAnswering, self).__init__(config)

        self.transformer = XLMModel(config)
        self.qa_outputs = SQuADHead(config)

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        lengths=None,
        position_ids=None,
        langs=None,
        token_type_ids=None,
        attention_mask=None,
        cache=None,
        start_positions=None,
        end_positions=None,
        cls_index=None,
        is_impossible=None,
        p_mask=None,
        head_mask=None,
        vat_configs=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            lengths=lengths,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            langs=langs,
            attention_mask=attention_mask,
            cache=cache,
            head_mask=head_mask,
        )

        output = transformer_outputs[0]

        outputs = self.qa_outputs(
            output,
            start_positions=start_positions,
            end_positions=end_positions,
            cls_index=cls_index,
            is_impossible=is_impossible,
            p_mask=p_mask,
        )

        outputs = (
            outputs + transformer_outputs[1:]
        )  # Keep new_mems and attention/hidden states if they are here

        return outputs
