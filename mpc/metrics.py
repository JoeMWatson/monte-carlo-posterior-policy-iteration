"""
Metrics used for MPC experiments.
"""

import numpy as np
from scipy.fftpack import fft


def fft_smoothness(
    action_sequence: np.ndarray, dt: float
) -> (float, float, np.ndarray, np.ndarray, np.ndarray):
    """
    Computes a smoothness measure for a vector-valued signal.
    see Regularizing Action Policies for Smooth Control with Reinforcement Learning

    Returns:
        Smoothness metric over vector
        Largest smoothness metric over signal indices
        FFT amplitude and frequencies
        Normalized signal used for metric
    """
    n, d = action_sequence.shape
    freq = np.linspace(0, 0.5 / dt, n // 2)

    def smoothness(signal):
        """Computes FFT and smoothness metric."""
        sp = 2 * np.abs(fft(signal)[: n // 2]) / n
        sm = 2 * np.einsum("n,n->", sp, freq)
        return sm, sp

    sm_max = 0
    for idx in range(d):
        sm_, _ = smoothness(action_sequence[:, idx])
        sm_max = sm_ if sm_ > sm_max else sm_max

    action_sequence_norm = np.linalg.norm(action_sequence, axis=1)

    sm, sp = smoothness(action_sequence_norm)

    return sm, sm_max, sp, freq, action_sequence_norm


def signal_power(action_sequence: np.ndarray):
    """Average L2 norm of vector-valued signal.

    Returns scalar power metric.
    """
    action_sequence_sq = np.linalg.norm(action_sequence, axis=1)
    power = action_sequence_sq.mean()
    return power
