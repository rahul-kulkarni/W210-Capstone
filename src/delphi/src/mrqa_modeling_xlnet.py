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
from .huggingface.pytorch_transformers.modeling_utils import add_start_docstrings
from .huggingface.pytorch_transformers.modeling_xlnet import (
    XLNetPreTrainedModel,
    XLNetModel,
    PoolerStartLogits,
    PoolerEndLogits,
    PoolerAnswerClass,
    XLNET_START_DOCSTRING,
    XLNET_INPUTS_DOCSTRING,
)
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from .huggingface.pytorch_transformers.tokenization_xlnet import XLNetTokenizer
from .mrqa_modeling_utils import VATLoss, VATType, VATConfig


class MRQAXLNetTokenizer(XLNetTokenizer):
    def mrqa_tokenize_text(self, text):
        # replace_words = ["[TLE]", "[PAR]", "[DOC]"]  # TODO:? Yi: could this be the problem!? I think AllenNLP also did this commented out part
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
        output_tokens = []
        for token in tokens:
            output_tokens.extend(self.tokenize(token))
        return output_tokens


@add_start_docstrings(
    """XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    XLNET_START_DOCSTRING,
    XLNET_INPUTS_DOCSTRING,
)
class XLNetForMRQAQuestionAnswering(XLNetPreTrainedModel):
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
        **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...).
            1.0 means token should be masked. 0.0 mean token is not masked.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned if both ``start_positions`` and ``end_positions`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification losses.
        **start_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top)``
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        **start_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``
            Indices for the top config.start_n_top start token possibilities (beam-search).
        **end_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **end_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **cls_logits**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size,)``
            Log probabilities for the ``is_impossible`` label of the answers.
        **mems**:
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `mems` input above). Can be used to speed up sequential decoding and attend to longer context.
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
        super(XLNetForMRQAQuestionAnswering, self).__init__(config)
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.transformer = XLNetModel(config)
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        input_mask=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        start_positions=None,
        end_positions=None,
        cls_index=None,
        is_impossible=None,
        p_mask=None,
        head_mask=None,
        vat_configs: Dict[VATType, VATConfig] = None,
    ):
        if not vat_configs:
            vat_configs = {}
        transformer_outputs = self.transformer(
            input_ids,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            head_mask=head_mask,
        )
        hidden_states = transformer_outputs[0]
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        hidden_states_copy = hidden_states.clone().detach().requires_grad_(False)
        last_layer_vat_config = vat_configs.get(VATType.LAST_HIDDEN_LAYER)
        vat_multiplier = 0
        if last_layer_vat_config:

            vat_multiplier = last_layer_vat_config.vat_multiplier
            vat_loss_fn = VATLoss(last_layer_vat_config)
            vat_loss = vat_loss_fn(self.start_logits, x=hidden_states_copy, dim=0, p_mask=p_mask)
        else:
            vat_loss = 0

        outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(
                hidden_states, start_positions=start_positions, p_mask=p_mask
            )

            if last_layer_vat_config:
                vat_loss_fn = VATLoss(last_layer_vat_config)
                vat_loss += vat_loss_fn(
                    self.end_logits,
                    x=hidden_states_copy,
                    dim=0,
                    start_positions=start_positions,
                    p_mask=p_mask,
                )

            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            assert vat_loss >= 0
            total_loss = (1 - vat_multiplier) * total_loss + vat_multiplier * vat_loss

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(
                    hidden_states, start_positions=start_positions, cls_index=cls_index
                )
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5
            outputs = (total_loss,) + outputs

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(
                -1, -1, hsz
            )  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(
                hidden_states, -2, start_top_index_exp
            )  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(
                -1, slen, -1, -1
            )  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(
                hidden_states_expanded, start_states=start_states, p_mask=p_mask
            )
            end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = torch.einsum(
                "blh,bl->bh", hidden_states, start_log_probs
            )  # get the representation of START as weighted sum of hidden states
            cls_logits = self.answer_class(
                hidden_states, start_states=start_states, cls_index=cls_index
            )  # Shape (batch size,): one single `cls_logits` for each sample

            outputs = (
                start_top_log_probs,
                start_top_index,
                end_top_log_probs,
                end_top_index,
                cls_logits,
            ) + outputs

        # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits
        # or (if labels are provided) (total_loss,)
        return outputs
