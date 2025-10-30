"""Deep CORAL objective used in the Yuan-style, backbone-matched comparator.

This is an implementation of a published objective, not a new method or an
exact replication of the Hyser/VGG experiment. Target labels are absent from
the unsupervised objective's interface.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def coral_loss(source, target):
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Expected two feature matrices with equal feature width")
    if min(len(source), len(target)) < 2 or source.shape[1] == 0:
        raise ValueError("Covariance needs at least two observations per domain")
    a = source - source.mean(0, keepdim=True)
    b = target - target.mean(0, keepdim=True)
    cs = a.T @ a / (len(source) - 1)
    ct = b.T @ b / (len(target) - 1)
    return (cs - ct).square().sum() / (4 * source.shape[1] ** 2)


def source_alignment_objective(model, source_x, source_y, target_x, weight):
    """Source CE + weighted feature covariance alignment; no target labels."""
    if weight < 0:
        raise ValueError("Alignment weight must be nonnegative")
    sz = model.encoder(source_x)
    source_ce = F.cross_entropy(model.head(sz), source_y)
    if weight == 0:
        domain = source_ce.new_zeros(())
    else:
        domain = coral_loss(sz, model.encoder(target_x))
    return source_ce + weight * domain, source_ce, domain
