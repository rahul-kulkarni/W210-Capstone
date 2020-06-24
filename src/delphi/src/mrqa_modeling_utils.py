#
# File modified on August 9, 2019 by Apple team.
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#

from __future__ import absolute_import, division, print_function, unicode_literals

import contextlib
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F


class VATType(Enum):
    INPUT_EMBEDDING = 0
    LAST_HIDDEN_LAYER = 1


class VATConfig:
    def __init__(self, vat_multiplier=0, xi=10, eps=1.0, ip=1):
        self.vat_multiplier = vat_multiplier
        self.xi = xi
        self.eps = eps
        self.ip = ip


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):
    def __init__(self, vat_config: VATConfig = None):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        if vat_config is None:
            vat_config = VATConfig()
        self.xi = vat_config.xi
        self.eps = vat_config.eps
        self.ip = vat_config.ip

    def forward(self, encoder, x, dim=1, **kwargs):
        with torch.no_grad():
            pred = F.softmax(encoder(x, **kwargs)[-1], dim=dim)

        with _disable_tracking_bn_stats(encoder):
            # calc adversarial direction
            max_distance = None
            best_d = None
            for _ in range(self.ip):
                # prepare random unit tensor
                d = torch.rand(x.shape).sub(0.5).to(x.device)
                d = _l2_normalize(d)
                d.requires_grad_()
                pred_hat = encoder(x + self.xi * d, **kwargs)[-1]
                logp_hat = F.log_softmax(pred_hat, dim=dim)
                adv_distance = F.kl_div(logp_hat, pred, reduction="batchmean")

                if max_distance is None or adv_distance > max_distance:
                    max_distance = adv_distance
                    adv_distance.backward()
                    d = _l2_normalize(d.grad)
                    best_d = d
                encoder.zero_grad()

            # calc LDS
            r_adv = best_d * self.eps
            pred_hat = encoder(x + r_adv, **kwargs)[-1]
            logp_hat = F.log_softmax(pred_hat, dim=dim)
            lds = F.kl_div(logp_hat, pred, reduction="batchmean")

        return lds
