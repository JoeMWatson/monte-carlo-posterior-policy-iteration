from pathlib import Path

import autograd.numpy as anp
import gym
import matplotlib.pyplot as plt
import mj_envs  # required to register envs
import numpy as np
from autograd import jacobian
from scipy.optimize import check_grad, minimize

import ppi.policies as policies
from ppi.samplers import MonteCarlo

DIR = Path(__file__).parent.resolve()


def plot_policy_samples(policy, n_samples, d_viz=10, filename=None):
    actions, _ = policy(n_samples)
    d_s, d_t, d_a = actions.shape
    d = min(d_a, d_viz)
    fig, axs = plt.subplots(d_a, figsize=(12, 9))
    axs = [axs] if d == 1 else axs
    axs[0].set_title(Path(filename).parts[-2])
    for i, ax in enumerate(axs):
        ax.plot(actions[:, :, i].T, ".-", alpha=0.3)
    if filename:
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)


ENVS = {
    # "Hopper-v2": 240,
    # "HalfCheetah-v2": 240,
    "HumanoidStandup-v2": 30,
    "door-v0": 30,
    "hammer-v0": 30,
}

POLICIES = [
    # policies.Matern12PeriodicKernel,
    policies.SquaredExponentialKernel,
    # policies.Matern12Kernel,
    # policies.Matern32Kernel,
    # policies.Matern52Kernel,
    # policies.WhiteNoiseKernel,
]

RESULTS = {
    env_name: {policy_class.__name__: None for policy_class in POLICIES}
    for env_name in ENVS
}
MODELS = {
    env_name: {policy_class.__name__: None for policy_class in POLICIES}
    for env_name in ENVS
}

for env_name, eval_horizon in ENVS.items():
    # load moments
    moments = np.load(DIR / f"{env_name}_moments.npz")
    mean_target, cov_in_target, cov_out_target = map(
        moments.get, ["mean", "covariance_in", "covariance_out"]
    )
    mean_target = mean_target[:eval_horizon, :]
    cov_in_target = cov_in_target[:eval_horizon, :eval_horizon]
    env = gym.make(env_name)
    dt = env.unwrapped.dt
    horizon, d_action = mean_target.shape
    initial_time_sequence = dt * np.arange(0, horizon)
    isotropic = np.diag(cov_in_target).mean(0)
    print(isotropic, np.diag(cov_out_target)[:3])
    for policy_class in POLICIES:
        """
        TODO: what about fitting mean? input covariance scalar?
        """
        # optimization requires good initial guess it seems
        init_period = 100.0 * dt if env_name == "Hopper-v2" else 5.0 * dt
        mean_vec = mean_target.mean(0)
        # fit KL via hyperparameters
        policy = policy_class(
            time_sequence=initial_time_sequence,
            action_dimension=env.action_space.shape[0],
            mean=mean_vec,
            covariance_in=np.atleast_1d(isotropic),
            covariance_out=cov_out_target,
            # lengthscale=0.5 * dt,
            lengthscale=dt,
            period=init_period,
            sampler=MonteCarlo,
        )
        policy_name = policy_class.__name__
        print(policy_name)
        print(policy.param)
        # cov_in_target = policy.covariance_in.copy()

        kl_init = policies.matrix_gaussian_kl(
            policy.mean,
            policy.covariance_in,
            policy.covariance_out,
            mean_target,
            cov_in_target,
            cov_out_target,
        )
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].set_title(f"Input Covariance Target")
        ax[0].imshow(cov_in_target, interpolation="none")
        ax[1].set_title(f"{type(policy).__name__}")
        ax[1].imshow(policy.covariance_in, interpolation="none")
        plt.savefig(
            DIR / f"{env_name}_{type(policy).__name__}_covariance_in_init",
            bbox_inches="tight",
        )
        plt.close(fig)

        EPS = np.finfo(np.float64).tiny

        def multivariate_gaussian_kl(mu_1, sigma_1, mu_2, sigma_2):
            d = sigma_1.shape[0]
            diff = mu_2 - mu_1
            return 0.5 * (
                anp.log(max(EPS, anp.linalg.det(sigma_2)))
                - anp.log(max(EPS, anp.linalg.det(sigma_1)))
                + anp.trace(anp.linalg.solve(sigma_2, sigma_1))
                + diff @ anp.linalg.solve(sigma_2, diff)
                - d
            )

        def objective(param):
            covariance_in_ = policy._k(policy.t, policy.t, *param, lib=anp)
            kl = multivariate_gaussian_kl(
                np.zeros((horizon,)),
                cov_in_target,
                np.zeros((horizon,)),
                covariance_in_,
            )
            return kl

        objective_jac = jacobian(objective)

        # params_init = policy.param
        params_init = np.ones_like(policy.param)
        res = minimize(
            objective,
            x0=params_init,
            method="L-BFGS-B",
            jac=objective_jac,
            bounds=policy.param_bounds,
        )
        param_opt = res.x
        print(env_name, "opt param", param_opt)
        kl = res.fun
        nit = res.nit
        covariance_in = policy._k(policy.t, policy.t, *param_opt)
        policy.covariance_in = covariance_in
        policy.covariance_in_sqrt = np.linalg.cholesky(covariance_in)

        plot_policy_samples(
            policy, 1, filename=DIR / f"{env_name}_{policy_name}_opt_samples"
        )

        kl = policies.matrix_gaussian_kl(
            policy.mean,
            covariance_in,
            policy.covariance_out,
            mean_target,
            cov_in_target,
            cov_out_target,
        )
        print(kl)
        RESULTS[env_name][policy_name] = kl
        MODELS[env_name][policy_name] = {
            "mean": mean_vec,
            "covariance_out": cov_out_target,
            "param": param_opt,
        }
        print(
            f"{env_name}, {policy_class}: {kl_init}->{kl} after {nit} iterations | {params_init} -> {param_opt}"
        )
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].set_title(f"Input Covariance Target")
        ax[0].imshow(cov_in_target, interpolation="none")
        ax[1].set_title(f"{policy_name}")
        ax[1].imshow(covariance_in, interpolation="none")
        # bar = plt.colorbar(shw)
        plt.savefig(
            DIR / f"{env_name}_{policy_name}_covariance_in", bbox_inches="tight"
        )
        plt.close(fig)
        # plt.show()
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].set_title(f"Output Covariance Target")
        ax[0].imshow(cov_out_target, interpolation="none")
        ax[1].set_title(f"{type(policy).__name__}")
        ax[1].imshow(policy.covariance_out, interpolation="none")
        plt.savefig(
            DIR / f"{env_name}_{type(policy).__name__}_covariance_out",
            bbox_inches="tight",
        )
        plt.close(fig)

for env, policies in RESULTS.items():
    for policy, kl in policies.items():
        print(env, policy, kl)


np.savez(DIR / f"model_selection.npz", **MODELS)
