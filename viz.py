from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

max_rows = 10


def plot_algorithm_result(res, filename=None):
    n_k = len(res)
    fig, axs = plt.subplots(1, n_k, figsize=(21, 9))
    for ax, (k, v) in zip(axs, res.items()):
        ax.set_title(k)
        ax.plot(v, ".-")
    if filename:
        axs[0].set_title(Path(filename).parts[-2])
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)


def plot_mean_std_1d(mean, std, filename):
    fig, ax = plt.subplots(2)
    ax[0].plot(mean, "k")
    ax[0].plot(mean + std, "k--")
    ax[0].plot(mean - std, "k--")
    # ax[1].plot(ent, "r")
    if filename:
        ax[0].set_title(Path(filename).parts[-2])
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)


def plot_samples(samples, filename=None):
    fig, ax = plt.subplots()
    ax.plot(samples, "k", alpha=0.1)
    if filename:
        ax.set_title(Path(filename).parts[-2])
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)


def plot_policy_samples(policy, n_samples, d_viz=6, filename=None):
    actions, _ = policy(n_samples)
    d_s, d_t, d_a = actions.shape
    d = min(d_a, d_viz)
    fig, axs = plt.subplots(d, figsize=(12, 9))
    axs = [axs] if d == 1 else axs
    for i, ax in enumerate(axs):
        ax.plot(actions[:, :, i].T, ".-", alpha=0.3)
    if filename:
        axs[0].set_title(Path(filename).parts[-2])
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)


def plot_sequence(sequence, d_viz=10, filename=None):
    d_t, d_s = sequence.shape
    if d_viz is None:
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.plot(sequence)
        axs = [ax]
    else:
        d = min(d_s, d_viz)
        fig, axs = plt.subplots(d, figsize=(12, 9))
        axs = [axs] if d == 1 else axs
        for i, ax in enumerate(axs):
            ax.plot(sequence[:, i], ".-")
    if filename:
        axs[0].set_title(Path(filename).parts[-2])
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)


def plot_sequence_history(sequence, sequence_history, d_viz=20, filename=None):
    (d_t_,) = sequence.shape
    d_t, d_s, d_p = sequence_history.shape
    assert d_t == d_t_
    d_s = min(d_s, d_viz)
    fig, ax = plt.subplots(figsize=(12, 9))
    for t in range(d_t):
        for i in range(d_s):
            ax.plot(np.arange(t, t + d_p), sequence_history[t, i, :], alpha=0.1)
    ax.plot(sequence, "k.-")
    if filename:
        ax.set_title(Path(filename).parts[-2])
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)


def plot_smoothness(spectrum, frequency, signal, filename=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 9))
    ax[0].plot(signal)
    ax[1].plot(frequency, spectrum)
    ax[0].set_xlabel("Timesteps")
    ax[0].set_ylabel("Action Norm")
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Spectrum")
    if filename:
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
