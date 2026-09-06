"""Plain compact temporal CNN baseline; architecture is a development choice."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


class ChannelStandardizer:
    """One scale per channel, fitted to permitted training windows only."""
    def fit(self, windows):
        if windows.ndim != 3 or windows.shape[1] != 16 or not np.isfinite(windows).all():
            raise ValueError("Expected finite windows x 16 channels x samples")
        self.mean = windows.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
        self.scale = np.maximum(windows.std(axis=(0, 2), keepdims=True, dtype=np.float64), 1e-8)
        return self

    def transform(self, windows):
        if not hasattr(self, "mean"):
            raise ValueError("Fit training statistics before transformation")
        if windows.ndim != 3 or windows.shape[1] != 16 or not np.isfinite(windows).all():
            raise ValueError("Invalid EMG windows")
        return ((windows - self.mean) / self.scale).astype(np.float32)


class CompactEMGNet(nn.Module):
    def __init__(self, classes=17):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=4), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        # The full observation window is available at prediction time. The
        # convolutions do not make a samplewise streaming claim within it.
        return self.head(self.encoder(x))

    def set_update_mode(self, mode):
        if mode not in ("full", "head", "frozen"):
            raise ValueError("Mode must be full, head or frozen")
        for param in self.encoder.parameters():
            param.requires_grad_(mode == "full")
        for param in self.head.parameters():
            param.requires_grad_(mode in ("full", "head"))
