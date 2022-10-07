import logging
from functools import partial
from os.path import exists
from pathlib import Path

import envs
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal as mvn

import ppi.algorithms as algorithms
import ppi.policies as policies
import ppi.samplers as samplers
from utils import make_filepath, write_args

ALGORITHMS = algorithms.__all__
ENVS = envs.__all__
POLICIES = policies.__all__
SAMPLERS = samplers.__all__


def main(args):

    if args.dir is not None:
        filepath = make_filepath(
            Path(__file__).parent.resolve(),
            Path(args.dir)
            / f"{args.algorithm}_{args.env}_{args.policy}_{args.sampling}_{args.seed}_{args.name}",
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

    env_ = getattr(envs, args.env)
    policy_ = getattr(policies, args.policy)
    agent_ = getattr(algorithms, args.algorithm)
    sampler_ = getattr(samplers, args.sampling)

    np.random.seed(args.seed)

    env = env_()

    policy = policy_(
        time_sequence=env.t,
        action_dimension=env.dim_action,
        mean=env.action_0,
        covariance_in=np.array([1e2]),  # 5e1
        covariance_out=np.diag([1e-3, 1e-3]),
        lengthscale=np.sqrt(3e-2),
        n_features=20,
        order=10,
        sampler=sampler_,
        use_derivatives=True,
        add_bias=True,
    )
    if env.condition:
        t0, a0 = np.zeros((1,)), env.action_0[None, :]
        policy.condition(t0, a0)

    agent = agent_(
        alpha=args.alpha,
        epsilon=args.epsilon,
        delta=args.delta,
        n_elites=args.n_elites,
        base_entropy=-200,
        entropy_rate=0.99,
        dimension=policy.dim_features,
    )

    res = agent(
        env,
        policy=policy,
        n_samples=args.n_samples,
        n_iters=args.n_iters,
        callback=partial(env.callback, path=None),
    )
    samples, _ = policy(25)
    env.callback(args.n_iters, None, samples, None, policy, filepath)
    logging.info(f"Success rate: {env.success_rate}")

    n_fields = len(res.keys())
    fig, axs = plt.subplots(1, n_fields, figsize=(12, 9))
    for i, (k, v) in enumerate(res.items()):
        if k in ["mean", "kl"]:
            axs[i].set_yscale("log")
        axs[i].plot(v, label=args.algorithm)
        axs[i].set_title(k)
        axs[i].legend()
    if filepath is not None:
        plt.savefig(filepath / "result.png", bbox_inches="tight")
        plt.close(fig)
        res["episodes"] = args.n_samples * np.arange(0, args.n_iters)
        res["success_rate"] = np.asarray(env.success_rate)
        np.savez(filepath / "data.npz", **res)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", choices=ALGORITHMS, default="Reps")
    parser.add_argument("env", choices=ENVS, default="BallInACup")
    parser.add_argument("policy", choices=POLICIES, default="RbfFeatures")
    parser.add_argument("--n-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--name", type=str, default="", help="name of experiment")
    parser.add_argument(
        "--force", action="store_true", help="Force experiment even if file exists"
    )
    # algorithm specific hyperparameters
    parser.add_argument("--n-elites", type=int, default=10, help="CEM elites")
    parser.add_argument(
        "--alpha", type=float, default=0.9, help="CEM smoothing on new values"
    )
    parser.add_argument(
        "--ess-pc", type=float, default=0.25, help="Effective Samples Size percentage"
    )
    parser.add_argument("--base-entropy", type=float, default=-100, help="MORE")
    parser.add_argument("--entropy-rate", type=float, default=0.99, help="MORE")
    parser.add_argument("--epsilon", type=float, default=1.0, help="KL bound")
    parser.add_argument(
        "--delta", type=float, default=1.0, help="Lower bound probability"
    )

    subparsers = parser.add_subparsers(title="sampling", dest="sampling")
    subparsers.required = True
    sample_parsers = [subparsers.add_parser(samp) for samp in SAMPLERS]
    for sp in sample_parsers:
        sp.add_argument("--n-samples", type=int, default=10)

    args_ = parser.parse_args()
    main(args_)
    plt.show()
