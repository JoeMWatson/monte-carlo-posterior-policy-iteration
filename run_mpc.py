import logging
from multiprocessing import cpu_count
from os.path import exists
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, cast

import dmc2gym
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
from mpc.metrics import fft_smoothness, signal_power
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
    plot_smoothness,
)

ALGORITHMS = algorithms.__all__
POLICIES = policies.__all__
SAMPLERS = samplers.__all__
ENVS = [
    "Hopper-v2",
    "Walker2d-v2",
    "HalfCheetah-v2",
    "HumanoidStandup-v2",
    "pen-v0",
    "relocate-v0",
    "door-v0",
    "hammer-v0",
    "walker~walk",
    "finger~spin",
    "FetchPickAndPlace-v1",
]

DIR = Path(__file__).parent.resolve()


def model_selection(env_name, policy):
    print(env_name, policy)
    models = np.load(
        DIR / "mpc" / "model_selection" / "model_selection.npz", allow_pickle=True
    )
    policy_moments = models[env_name].item()
    moments = policy_moments[policy]
    covar_in = np.array([1.0])  # is rather obtained as kernel params
    return moments["mean"], covar_in, moments["covariance_out"], moments["param"]


def get_state(env):
    return MujocoEnvHandler.get_current_state(cast(gym.wrappers.TimeLimit, env))


def get_control_timestep(env):
    if hasattr(env.unwrapped, "dt"):
        return env.unwrapped.dt  # gym
    else:
        return env.unwrapped._env.control_timestep()  # dmcs


def save_frames_as_gif(frames, filename="gym_animation", fps=60):

    # mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(f"{filename}.mp4", writer=writer)
    plt.close("all")


def main(args):

    if args.dir is not None:
        filepath = make_filepath(
            DIR,
            Path("mpc")
            / Path(args.dir)
            / f"{args.algorithm}_{args.env}_{args.policy}_{args.sampling}_{args.n_samples}_{args.seed}_{args.name}",
            filename=None,
        )
        make_full_path = lambda path: filepath / path
        if exists(make_full_path("data.npz")):
            if args.force:
                print("experiment done, but repeating")
            else:
                print("experiment done!")
                exit()
        write_args(args, filepath)
        logging.basicConfig(
            handlers=[
                logging.FileHandler(filename=filepath / "log", mode="w"),
                logging.StreamHandler(),
            ],
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
        )
        for k, v in vars(args).items():
            logging.info(f"{k} = {v}")
    else:
        filepath = None
        make_full_path = lambda path: None

    if "~" in args.env:
        domain, task = args.env.split("~")
        env = dmc2gym.make(domain_name=domain, task_name=task)
    else:
        env = gym.make(args.env)

    np.random.seed(args.seed)
    env.seed(args.seed)

    policy_ = getattr(policies, args.policy)
    solver_ = getattr(algorithms, args.algorithm)
    sampler_ = getattr(samplers, args.sampling)

    mean, covariance_in, covariance_out = policies.design_moments(
        env.action_space.low, env.action_space.high, ratio=1000.0
    )

    dt = get_control_timestep(env)
    total_time_sequence = (
        dt * np.arange(0, args.timesteps)
        if args.policy == "RbfFeatures"
        else dt * np.arange(0, args.horizon)
    )  # used for RBF construction
    limiter = policies.Limiter(lower=env.action_space.low, upper=env.action_space.high)

    policy = policy_(
        time_sequence=total_time_sequence,
        action_dimension=env.action_space.shape[0],
        mean=mean,
        covariance_in=covariance_in,
        covariance_out=covariance_out,
        lengthscale=args.lengthscale,
        period=dt,
        n_features=args.n_features,
        order=args.order,
        sampler=sampler_,
        beta=args.beta,
        limiter=limiter,
        use_derivatives=False,
        use_bias=True,
    )

    if not args.no_plots:
        initial_policy_samples, _ = policy(6)
        plot_policy_samples(policy, 6, filename=make_full_path("policy_prior"))

    solver = solver_(
        alpha=args.alpha,
        epsilon=args.epsilon,
        delta=args.delta,
        n_elites=args.n_elites,
        dimension=policy.dim_features,
    )

    env_mpc = ControlEnv(args.env, dt, args.n_processes)
    agent = Mpc(
        env_mpc,
        dt,
        solver,
        policy,
        args.timesteps,
        args.horizon,
        args.n_samples,
        args.n_iters,
        args.anneal,
        use_map="iCem" in args.algorithm,  # Cem or iCem
    )

    env.reset()
    state = get_state(env)

    if args.n_warmstart_iters > 0:
        mean_cost, std_cost, res = agent.warm_start(
            state,
            time_index=0,
            n_iters=args.n_warmstart_iters,
            # callback=callback
        )
        logging.info(f"Warm start: {mean_cost[-1]:.2f} +/- {std_cost[-1]:.2f}")

        if not args.no_plots:
            plot_algorithm_result(res, make_full_path("result_warmup"))
            plot_mean_std_1d(mean_cost, std_cost, make_full_path("solver_warmup"))
            plot_samples(agent.rewards_warmstart.T, make_full_path("rewards_warmup"))
            plot_policy_samples(policy, 6, filename=make_full_path("policy_warmup"))

    frames = []

    if isinstance(env.observation_space, gym.spaces.Dict):
        dim_obs = sum([y.shape[0] for y in env.observation_space.spaces.values()])
        process_obs = lambda y: np.concatenate([y_ for y_ in y.values()])
    else:
        dim_obs = env.observation_space.shape[0]
        process_obs = lambda y: y
    obs = np.nan * np.ones((args.timesteps, dim_obs))
    acts = np.nan * np.ones((args.timesteps, env.action_space.shape[0]))
    rewards = np.nan * np.ones((args.timesteps,))
    ret = 0.0
    t = 0
    done = False
    try:
        time_range = range(args.timesteps)
        time_iter = time_range if args.no_tqdm else tqdm(time_range)
        for t in time_iter:
            np.random.seed(args.seed + t)
            # Render to frames buffer
            frames += [env.render(mode="rgb_array"),] if args.render else []
            state = get_state(env)
            # action = env.action_space.sample()
            action = agent(state, t)
            y, r, done, _ = env.step(action)
            ret += r
            obs[t, :], acts[t, :], rewards[t] = process_obs(y), action, r

            if (t == (args.timesteps - args.horizon)) and not args.no_plots:
                plot_policy_samples(
                    policy, 6, filename=make_full_path("policy_mid_episode")
                )

    except Exception:
        logging.exception("Experiment failed")
        import traceback

        print(traceback.format_exc())
    finally:
        env.close()

    logging.info(f"Done={done}, Return: {ret} after {t}/{args.timesteps} timesteps")

    if not args.no_plots:
        plot_sequence(obs, filename=make_full_path("observation_sequence"))
        plot_sequence(acts, d_viz=8, filename=make_full_path("action_sequence_8"))
        plot_sequence(acts, d_viz=None, filename=make_full_path("action_sequence_all"))
        plot_sequence(agent.ess, filename=make_full_path("ess_history"))
        plot_sequence(agent.alphas, filename=make_full_path("alpha_history"))
        plot_sequence_history(
            rewards, agent.rewards, filename=make_full_path("reward_sequence")
        )

    power = signal_power(acts)
    sm, sm_max, sp, freq, action_sequence_norm = fft_smoothness(acts, dt)
    logging.info(f"Smoothness: {sm:.3f}, Max: {sm_max:.3f}")
    if not args.no_plots:
        plot_smoothness(
            sp, freq, action_sequence_norm, filename=make_full_path("smoothness")
        )

    if args.render and filepath is not None:
        fps = int(1 / dt)
        print(f"Saving frames at {fps} FPS")
        save_frames_as_gif(frames, filepath / args.env, fps=fps)

    if filepath is not None:
        res = {
            "obs": obs,
            "actions": acts,
            "rewards": rewards,
            "ess": agent.ess,
            "sm": sm,
            "sm_max": sm_max,
            "power": power,
            "action_signal": action_sequence_norm,
        }
        np.savez(filepath / "data.npz", **res)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", choices=ALGORITHMS, default="Reps")
    parser.add_argument("env", choices=ENVS, default="BallInACup")
    parser.add_argument("policy", choices=POLICIES, default="RbfFeatures")
    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--n-warmstart-iters", type=int, default=50)
    parser.add_argument("--n-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-processes", type=int, default=cpu_count())
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--name", type=str, default="", help="name of experiment")
    parser.add_argument(
        "--force", action="store_true", help="Force experiment even if file exists"
    )
    parser.add_argument(
        "--anneal", type=float, default=1.0, help="Anneal policy update"
    )
    # algorithm specific hyperparameters
    parser.add_argument("--n-elites", type=int, default=10, help="CEM elites")
    parser.add_argument("--alpha", type=float, default=10, help="AIS temperature")
    parser.add_argument("--epsilon", type=float, default=2.0, help="KL bound")
    parser.add_argument("--delta", type=float, default=0.9, help="Lower bound")
    # policy specific hyperparameters
    parser.add_argument("--beta", type=float, default=2.0, help="Coloured Noise")
    parser.add_argument(
        "--lengthscale", type=float, default=1.0, help="Kernel lengthscale"
    )
    parser.add_argument(
        "--n-features", type=int, default=10, help="Number of RBF features"
    )
    parser.add_argument(
        "--order", type=int, default=10, help="Order of QRFF, features are x2"
    )

    subparsers = parser.add_subparsers(title="sampling", dest="sampling")
    subparsers.required = True
    sample_parsers = [subparsers.add_parser(samp) for samp in SAMPLERS]
    for sp in sample_parsers:
        sp.add_argument("--n-samples", type=int, default=10)

    args_ = parser.parse_args()
    main(args_)
    plt.show()
