from pathlib import Path

import d4rl  # Import required to register environments
import gym
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from scipy.fftpack import fft


def fft_smoothness(action_sequence, dt, norm=True):
    """
    Regularizing Action Policies for Smooth Control with Reinforcement Learning
    """
    n, d = action_sequence.shape
    if norm:
        action_sequence_norm = np.linalg.norm(action_sequence, axis=1)
    else:
        assert d == 1, d
        action_sequence_norm = action_sequence[:, 0]
    sp = 2 * np.abs(fft(action_sequence_norm)[: n // 2]) / n
    freq = np.linspace(0, 0.5 / dt, n // 2)
    sm = 2 * np.einsum("n,n->", sp, freq)
    return sm, sp, freq, action_sequence_norm


import ppi.policies as policies

DIR = Path(__file__).parent.resolve()

ENVS = {
    # "HumanoidStandup-v2": "cem",
    # "HumanoidStandup-v2": "gac",
    # "Hopper-v2": "hopper-expert-v2",
    # "HalfCheetah-v2": "halfcheetah-expert-v2",
    # "door-v0": "door-expert-v1",
    # "hammer-v0": "hammer-expert-v1",
    "hammer-v0": "se",
    "HumanoidStandup-v2": "se",
    # "HumanoidStandup-v2": "gac",
    # "Hopper-v2": "hopper-expert-v2",
    # "HalfCheetah-v2": "halfcheetah-expert-v2",
    # "door-v0": "door-expert-v1",
    # "hammer-v0": "hammer-expert-v1",
}

n_batches = 500
horizon = 250
episode_length = 1000
horizon = 250
# policies_ = [
#     policies.SquaredExponentialKernel,
#     policies.Matern12PeriodicKernel,
#     policies.WhiteNoiseKernel,
# ]
for env_name, dataset_name in ENVS.items():
    env_real = gym.make(env_name)
    limiter = policies.Limiter(env_real.action_space.low, env_real.action_space.high)
    if "HumanoidStandup-v2" == env_name:
        if dataset_name == "cem":
            a = np.load(DIR / "data" / "HumanoidStandup-v2_cem_mpc_data.npz")
            actions = limiter(a["actions"][:50, :horizon, :])
            n_b, _, d_a = actions.shape
            rewards = a["rewards"][:, 0]
            print(actions.shape)
            print(rewards.shape)
        elif dataset_name == "se":
            data = np.load(DIR / "data" / "HumanoidStandup-v2-se.npz")
            actions = data["actions"].transpose((2, 0, 1))

            rewards = data["rewards"][:, :].T
            actions = actions[:, :horizon, :]
            n_b, _, d_a = actions.shape
            import pdb

            pdb.set_trace()
            print(actions.shape)
            print(rewards.shape)
        elif dataset_name == "gac":
            a = np.load(DIR / "data" / "HumanoidStandupEnv_data.npz")
            actions = a["actions"][:, :horizon, :]
            rewards = a["rewards"][:, :horizon, 0]
            n_b, _, d_a = actions.shape
            print(actions.shape)
            print(rewards.shape)
        else:
            raise ValueError()
    elif env_name == "hammer-v0":
        if dataset_name == "se":

            data = np.load(DIR / "data" / "hammer-v0-se.npz")
            actions = data["actions"].transpose((2, 0, 1))
            rewards = data["rewards"][:, :].T
            actions = actions[:, :horizon, :]
            n_b, _, d_a = actions.shape
            import pdb

            pdb.set_trace()
            print(actions.shape)
            print(rewards.shape)
    else:
        env = gym.make(dataset_name)
        dataset = env.get_dataset()
        terminals = dataset["terminals"]
        # actions_flat = limiter()
        actions_flat = dataset["actions"]
        rewards_flat = dataset["rewards"]
        n, d_a = actions_flat.shape
        n_b = n // episode_length
        print(n_b)
        actions = np.zeros((n_b, horizon, d_a))
        rewards = np.zeros((n_b, horizon))
        for i in range(n_b):
            idx = episode_length * i
            rewards[i, :] = rewards_flat[idx : idx + horizon]
            actions[i, :, :] = actions_flat[idx : idx + horizon, :]
        print(actions.shape)

    dt = env_real.unwrapped.dt

    rets = rewards[:, :horizon].sum(axis=1)
    ret_med = np.median(rets, axis=0)
    print(f"Median_reward = {ret_med}")
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(env_name)
    ax.plot(rewards[:10, :horizon].T)
    plt.savefig(DIR / f"{env_name}_rewards_horizon", bbox_inches="tight")
    # plt.close(fig)

    lower, mid, upper = np.percentile(rets, [25, 50, 75], axis=0)
    print("rets", lower, mid, upper)

    # import pdb; pdb.set_trace()
    smoothness = np.zeros((n_b))
    norm = np.zeros((250, n_b))
    for i in range(n_b):
        sm, sp, freq, action_sequence_norm = fft_smoothness(actions[i, :, :], dt)
        smoothness[i] = sm
        norm[:, i] = action_sequence_norm

    lower, mid, upper = np.percentile(smoothness, [25, 50, 75], axis=0)
    print("smoothness", lower, mid, upper)
    for i in range(d_a):
        print(
            "smoothness 0 0", fft_smoothness(actions[0, :, i, None], dt, norm=False)[0]
        )

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(env_name)
    ax.plot(norm[:, :])
    plt.savefig(DIR / f"{env_name}_smoothnorm", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(env_name)
    ax.plot(rewards[:10, :].T)
    plt.savefig(DIR / f"{env_name}_rewards_all", bbox_inches="tight")
    # plt.close(fig)

    d = min(d_a, 6)
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.set_title(env_name)
    ax.plot(actions[0, :horizon, :], alpha=0.5)
    plt.savefig(DIR / f"{env_name}_samples", bbox_inches="tight")
    tikzplotlib.save(f"{env_name}_{dataset_name}_action_sequence", figure=fig)
    # plt.close(fig)

    action_data_vec = policies.vec(actions)
    cov_vec = np.cov(action_data_vec.T)

    plt.figure(figsize=(10, 10))
    plt.imshow(
        cov_vec[: 3 * horizon, : 3 * horizon], interpolation="none",
    )
    plt.savefig(DIR / f"{env_name}_vec_covariance_subset", bbox_inches="tight")

    mean, covariance_in, covariance_out, ess = policies.m_projection_mavn(
        np.zeros((n_b,)),
        actions[:n_b, :, :],
        np.eye(horizon),
        np.eye(d_a),
        iterations=5,
        update_out=True,
        use_tqdm=True,
    )
    print(mean[0, :])
    print(covariance_in[:6, :6])
    print(np.diag(covariance_out))

    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
    axs[0].imshow(covariance_in, interpolation="none")
    axs[0].legend()
    axs[0].set_title("$\\mathbf{K}$")
    axs[0].set_xlabel("$t$")
    axs[0].set_ylabel("$t$")
    # plt.savefig(DIR / f"{env_name}_covariance_in", bbox_inches="tight")
    axs[1].set_title("$\\mathbf{\Sigma}$")
    axs[1].imshow(covariance_out, interpolation="none")
    axs[1].set_xlabel("$a$")
    axs[1].set_ylabel("$a$")
    axs[2].plot(actions[0, :horizon, :], alpha=0.1, color="k")
    axs[2].set_xlim(0, 250)
    axs[2].set_title(env_name)
    axs[2].set_ylabel("$\\mathbf{a}$")
    axs[2].set_xlabel("Timesteps")
    plt.savefig(DIR / f"{env_name}_mavn", bbox_inches="tight")
    tikzplotlib.save(f"{env_name}_{dataset_name}_mavn.tex", figure=fig)
    #
    res = {
        "mean": mean,
        "covariance_in": covariance_in,
        "covariance_out": covariance_out,
    }
    np.savez(DIR / f"{env_name}_moments.npz", **res)

plt.show()
