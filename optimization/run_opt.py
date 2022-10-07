import logging
import pathlib
from os import makedirs
from os.path import exists, join

import functions
import matplotlib.pyplot as plt
import numpy as np

import ppi.algorithms as algorithms
from ppi.policies import GaussianPolicy
from ppi.samplers import CubatureQuadrature, MonteCarlo, QuasiMonteCarlo

ALGORITHMS = algorithms.__all__
FUNCTIONS = functions.__all__


def main(args):

    if args.dir is not None:
        dir = pathlib.Path(__file__).parent.resolve() / args.dir
        if not exists(dir):
            makedirs(dir)
        filepath = (
            dir
            / f"{args.algorithm}_{args.function}_{args.sampling}_{args.seed}_{args.name}"
        )
        print(filepath)
        if exists(f"{filepath}.npz") and not args.force:
            print("File exists!")
            return 0
        logging.basicConfig(
            handlers=[
                logging.FileHandler(filename=f"{filepath}.log", mode="w"),
                logging.StreamHandler(),
            ],
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
        )
        for k, v in vars(args).items():
            logging.info(f"{k} = {v}")

    np.random.seed(args.seed)
    alg = getattr(algorithms, args.algorithm)
    func = getattr(functions, args.function)
    solver = alg(**vars(args))

    function = func(args.dimension)
    if args.sampling == "mc":
        sampler = MonteCarlo(args.dimension)
    elif args.sampling == "qmc":
        sampler = QuasiMonteCarlo(args.dimension)
    elif args.sampling == "quad":
        sampler = CubatureQuadrature(args.dimension)

    policy = GaussianPolicy(
        mu=np.ones((args.dimension,)),
        sigma=0.5 * np.eye(args.dimension),
        sampler=sampler,
        diagonal=args.algorithm == "Cem",
    )
    try:
        res = solver(function, policy, args.n_samples, args.n_iter)
    except Exception:
        logging.exception("Experiment failed")
        import traceback

        print(traceback.format_exc())
        exit()

    if args.dir is not None:
        res["episodes"] = args.n_samples * np.arange(0, args.n_iter)
        np.savez(f"{filepath}.npz", **res)
    else:
        filepath = None

    if args.plot:
        n_fields = len(res.keys())
        fig, axs = plt.subplots(1, n_fields, figsize=(21, 9))
        for i, (k, v) in enumerate(res.items()):
            if k in ["mean", "kl"]:
                axs[i].set_yscale("log")
            axs[i].plot(v, label=args.algorithm)
            axs[i].set_title(k)
            axs[i].legend()
            if filepath is not None:
                plt.savefig(f"{filepath}.png", bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", choices=ALGORITHMS)
    parser.add_argument("function", choices=FUNCTIONS)
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument(
        "--force", action="store_true", help="Force experiment even if file exists"
    )
    # algorithm specific hyperparameters
    parser.add_argument("--n-elites", type=int, default=10, help="CEM elites")
    parser.add_argument(
        "--alpha", type=float, default=0.9, help="CEM smoothing on new values"
    )
    parser.add_argument("--base-entropy", type=float, default=-100, help="MORE")
    parser.add_argument("--entropy-rate", type=float, default=0.99, help="MORE")
    parser.add_argument("--epsilon", type=float, default=0.1, help="KL bound")
    parser.add_argument(
        "--delta", type=float, default=0.5, help="Lower bound probability"
    )
    parser.add_argument(
        "--ess-pc", type=float, default=0.1, help="Effective sample size percentage"
    )

    subparsers = parser.add_subparsers(title="sampling", dest="sampling")
    subparsers.required = True
    parser_mc = subparsers.add_parser("mc", help="Monte Carlo sampling")
    parser_mc.add_argument("--n-samples", type=int, default=100)
    parser_qmc = subparsers.add_parser("qmc", help="Quasi Monte Carlo")
    parser_qmc.add_argument("--n-samples", type=int, default=100)
    parser_quad = subparsers.add_parser("quad", help="Sparse cubature quadrature")

    args_ = parser.parse_args()
    main(args_)
