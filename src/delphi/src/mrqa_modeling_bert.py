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

from typing import Dict

import torch
from .huggingface.pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)
from .huggingface.pytorch_transformers.modeling_utils import add_start_docstrings
from .huggingface.pytorch_transformers.tokenization_bert import BertTokenizer

from torch import nn
from torch.nn import CrossEntropyLoss

from .mrqa_modeling_utils import VATLoss, VATConfig, VATType


class MRQABertTokenizer(BertTokenizer):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def mrqa_tokenize_text(self, text):
        # replace_words = ["[TLE]", "[PAR]", "[DOC]"]  # TODO:? Yi: could this be the problem!?
        # for word in replace_words:
        #     text = text.replace(word, "[SEP]")
        tokenized_text = self.tokenize(text)
        return tokenized_text

    def mrqa_tokenize_tokens(self, tokens):
        # replace_words = ["[TLE]", "[PAR]", "[DOC]"]
        # replaced_tokens = []
        # for token in tokens:
        #     if token in replace_words:
        #         token = "[SEP]"
        #     replaced_tokens.append(token)
        # TODO: Alternatively, join these tokens with " " and prcess them altogether.
        # [yi]: don't join them with space. MRQA did subword tokenization more than just whitespace tokenizatio
        output_tokens = []
        for token in tokens:
            output_tokens.extend(self.mrqa_tokenize(token))
        return output_tokens

@add_start_docstrings(
    """Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)
class BertForMRQAQuestionAnswering(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

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

        >>> config = BertConfig.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>>
        >>> model = BertForQuestionAnswering(config)
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> start_positions = torch.tensor([1])
        >>> end_positions = torch.tensor([3])
        >>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        >>> loss, start_scores, end_scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForMRQAQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertMRQAModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
        position_ids=None,
        head_mask=None,
        vat_configs: Dict[VATType, VATConfig] = None,
    ):
        if not vat_configs:
            vat_configs = {}
        non_vat_multiplier = 1
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            vat_config=vat_configs.get(VATType.INPUT_EMBEDDING),
        )

        if vat_configs is not None:
            for k, v in vat_configs.items():
                non_vat_multiplier -= v.vat_multiplier

        assert 0.5 < non_vat_multiplier <= 1
        sequence_output = outputs[0]
        input_vat_loss = outputs[2]

        logits = self.qa_outputs(sequence_output)
        last_layer_vat_config = vat_configs.get(VATType.LAST_HIDDEN_LAYER)
        if last_layer_vat_config:
            last_layer_vat_loss_fn = VATLoss(last_layer_vat_config)
            sequence_output_copy = sequence_output.clone().detach().requires_grad_(False)
            last_layer_vat_loss = last_layer_vat_loss_fn(self.qa_outputs, x=sequence_output_copy)
        else:
            last_layer_vat_loss = 0

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits) + outputs[3:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if non_vat_multiplier > 0:
                total_loss = (
                    non_vat_multiplier * total_loss
                    + vat_configs.get(VATType.INPUT_EMBEDDING, VATConfig()).vat_multiplier
                    * input_vat_loss
                    + vat_configs.get(VATType.LAST_HIDDEN_LAYER, VATConfig()).vat_multiplier
                    * last_layer_vat_loss
                )
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions), vat_loss


class BertMRQAModel(BertModel):
    def __init__(self, config):
        super(BertMRQAModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        vat_config: VATConfig = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        embedding_output_copy = embedding_output.clone().detach().requires_grad_(False)

        if vat_config:
            vat_loss_fn = VATLoss(vat_config)
            vat_loss = vat_loss_fn(
                self.encoder,
                x=embedding_output_copy,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
            )
        else:
            vat_loss = 0

        encoder_outputs = self.encoder(
            embedding_output, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (
            (sequence_output, pooled_output, vat_loss)
            + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        )

        return outputs  # sequence_output, pooled_output, vat_loss, (hidden_states), (attentions)
