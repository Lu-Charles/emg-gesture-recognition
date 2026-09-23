"""Deep CORAL objective used in the Yuan-style, backbone-matched comparator.

This is an implementation of a published objective, not a new method or an
exact replication of the Hyser/VGG experiment. Target labels are absent from
the unsupervised objective's interface.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def coral_loss(source, target):
    """Compare unbiased feature covariances, scaled by 4 times feature width squared."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Expected two feature matrices with equal feature width")
    if min(len(source), len(target)) < 2 or source.shape[1] == 0:
        raise ValueError("Covariance needs at least two observations per domain")
    source_centered = source - source.mean(0, keepdim=True)
    target_centered = target - target.mean(0, keepdim=True)
    # Use n - 1 in both domains, including when their batch sizes differ.
    source_covariance = source_centered.T @ source_centered / (len(source) - 1)
    target_covariance = target_centered.T @ target_centered / (len(target) - 1)
    return (source_covariance - target_covariance).square().sum() / (
        4 * source.shape[1] ** 2
    )


def source_alignment_objective(model, source_x, source_y, target_x, weight):
    """Source CE + weighted feature covariance alignment; no target labels."""
    if weight < 0:
        raise ValueError("Alignment weight must be nonnegative")
    source_features = model.encoder(source_x)
    source_ce = F.cross_entropy(model.head(source_features), source_y)
    # The source-only control must not depend on target samples.
    if weight == 0:
        alignment_loss = source_ce.new_zeros(())
    else:
        alignment_loss = coral_loss(source_features, model.encoder(target_x))
    return source_ce + weight * alignment_loss, source_ce, alignment_loss
