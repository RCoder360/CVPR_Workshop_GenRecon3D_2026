"""
Efficiency measurement utilities.

Logs: parameter count, peak VRAM, training time per iteration.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn


class EfficiencyTracker:
    """Track and report efficiency metrics during training."""

    def __init__(self):
        self._iter_times: list[float] = []
        self._t0: Optional[float] = None
        self._peak_mem: float = 0.0

    def start_iteration(self):
        """Call at the start of each training iteration."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def end_iteration(self):
        """Call at the end of each training iteration."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._t0 is not None:
            dt = time.perf_counter() - self._t0
            self._iter_times.append(dt)
        # Track peak memory
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            self._peak_mem = max(self._peak_mem, mem)

    def reset_peak_memory(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._peak_mem = 0.0

    # ------------------------------------------------------------------ #
    # Reports
    # ------------------------------------------------------------------ #
    @property
    def avg_iter_time(self) -> float:
        """Average iteration time in seconds."""
        if not self._iter_times:
            return 0.0
        return sum(self._iter_times) / len(self._iter_times)

    @property
    def total_time(self) -> float:
        return sum(self._iter_times)

    @property
    def peak_vram_mb(self) -> float:
        return self._peak_mem

    def report(self) -> dict[str, float]:
        return {
            "avg_iter_time_s": self.avg_iter_time,
            "total_train_time_s": self.total_time,
            "peak_vram_mb": self.peak_vram_mb,
        }


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(*args) -> int:
    """Count trainable parameters across multiple models and parameter lists."""
    total = 0
    for a in args:
        if isinstance(a, nn.Module):
            total += count_parameters(a)
        elif isinstance(a, torch.Tensor):
            if a.requires_grad:
                total += a.numel()
        elif isinstance(a, (list, tuple)):
            for p in a:
                if isinstance(p, torch.Tensor) and p.requires_grad:
                    total += p.numel()
                elif isinstance(p, nn.Module):
                    total += count_parameters(p)
    return total
