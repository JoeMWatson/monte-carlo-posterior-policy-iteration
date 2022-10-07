import logging
from multiprocessing import Pool, cpu_count
from os.path import exists
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, cast

import gym
import gym.wrappers
import matplotlib.pyplot as plt
import mj_envs  # required to register envs
import numpy as np
from matplotlib import animation
from tqdm import tqdm

import ppi.algorithms as algorithms
import ppi.policies as policies
import ppi.samplers as samplers
from mpc.mpc import Mpc
from mpc.wrappers import ControlEnv, MujocoEnvHandler
from utils import make_filepath, write_args
from viz import (
    plot_algorithm_result,
    plot_mean_std_1d,
    plot_policy_samples,
    plot_samples,
    plot_sequence,
    plot_sequence_history,
)

DIR = Path(__file__).parent.resolve()
ENV_NAME = "HumanoidStandup-v2"
TIMESTEPS = 250
HORIZON = 30
N_ITERS = 2
N_SAMPLES = 720
N_WARMSTART_ITERS = 50
N_EVALS = 200


def get_state(env):
    return MujocoEnvHandler.get_current_state(cast(gym.wrappers.TimeLimit, env))


def get_control_timestep(env):
    return env.unwrapped.dt  # gym


def main():

    env = gym.make(ENV_NAME)

    mean, covariance_in, covariance_out = policies.design_moments(
        env.action_space.low, env.action_space.high, ratio=1000.0
    )

    dt = get_control_timestep(env)
    initial_time_sequence = dt * np.arange(0, HORIZON)
    limiter = policies.Limiter(env.action_space.low, env.action_space.high)

    env.reset()
    state = get_state(env)

    obs = np.nan * np.ones((N_EVALS, TIMESTEPS, env.observation_space.shape[0]))
    acts = np.nan * np.ones((N_EVALS, TIMESTEPS, env.action_space.shape[0]))
    rewards = np.nan * np.ones((N_EVALS, TIMESTEPS,))

    done = False
    env_mpc = ControlEnv(ENV_NAME, dt, cpu_count())
    for k in range(N_EVALS):
        ret = 0.0
        t = 0
        policy = policies.WhiteNoiseIid(
            time_sequence=initial_time_sequence,
            action_dimension=env.action_space.shape[0],
            mean=mean,
            covariance_in=covariance_in,
            var_out=covariance_out,
            lengthscale=dt ** 2,
            period=dt,
            n_features=20,
            order=25,
            sampler=samplers.MonteCarlo,
            limiter=limiter,
        )

        solver = algorithms.Cem(
            n_elites_pc=0.1, entropy_rate=0.99, dimension=policy.dim_features,
        )

        agent = Mpc(
            env_mpc, dt, solver, policy, TIMESTEPS, HORIZON, N_SAMPLES, N_ITERS, 1.0,
        )
        np.random.seed(k)
        env.seed(k)
        env.reset()
        mean_cost, std_cost, res = agent.warm_start(
            state, time_index=0, n_iters=N_WARMSTART_ITERS,
        )
        print(f"{k:3d} | Warm start: {mean_cost[-1]:.2f} +/- {std_cost[-1]:.2f}")
        for t in tqdm(range(TIMESTEPS)):
            state = get_state(env)
            action = agent(state, t)
            y, r, done, _ = env.step(action)
            ret += r
            obs[k, t, :], acts[k, t, :], rewards[k, t] = y, action, r
            if done:
                break
        print(f"{k:3d} | Done={done}, Return: {ret} after {t}/{TIMESTEPS} timesteps")
    env.close()

    res = {"obs": obs, "actions": acts, "rewards": rewards}
    np.savez(DIR / f"{ENV_NAME}_cem_mpc_data.npz", **res)


if __name__ == "__main__":
    main()
