"""
A model predictive control API that wraps around PPI optimization and policies.
"""

from typing import Tuple

import numpy as np


class Mpc(object):
    """Model predictive control interface.

    Consists of
        env: see Mujoco wrapper
        solver: ppi.algorithms
        policy: ppi.policies
    """

    def __init__(
        self,
        env,
        dt: float,
        solver,
        policy,
        timesteps: np.ndarray,
        horizon: int,
        n_samples: int,
        n_iters: int,
        anneal: float,
        use_map=False,
    ):
        self.env = env
        self.dt = dt
        self.solver = solver
        self.policy = policy
        self.timesteps = timesteps
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.anneal = anneal
        self.use_map = use_map
        self.rewards = np.nan * np.ones((timesteps, n_samples, horizon))
        self.ess = np.nan * np.ones((timesteps, 1))
        self.alphas = np.nan * np.ones((timesteps, 1))
        self.rewards_warmstart = np.nan * np.ones((n_samples, horizon))

        plan_time = self.time_sequence(0)
        self.policy.compute_prior(plan_time)

    def time_sequence(self, time_index: int):
        """Make a new time horizon given current time index."""
        last_index = time_index + self.horizon
        if last_index > self.timesteps:
            last_index = self.timesteps

        return self.dt * np.arange(time_index, last_index)

    def optimize(self, state: Tuple, time_index: int, n_iters: int, use_tqdm: bool):
        """Optimize state trajectry about state and time_index for n_iters iterations.

        Returns:
            optimization trace result.
        """
        t = self.time_sequence(time_index)
        self.policy.update_timesteps(t, self.anneal)
        self.env.set_state(state)
        res = self.solver(
            self.env,
            policy=self.policy,
            n_samples=self.n_samples,
            n_iters=n_iters,
            use_tqdm=use_tqdm,
        )
        return res

    def __call__(self, state, time_index):
        """MPC optimization, called per timestep."""
        res = self.optimize(state, time_index, self.n_iters, use_tqdm=False)
        self.telemetry(time_index, res)
        if self.use_map:
            return self.policy.map_sequence[0, :]
        else:
            mean = self.policy.predict(only_mean=True)
            return mean[0, :]

    def telemetry(self, time_index, optimization_results):
        """Logging for performance analysis."""
        if "ess" in optimization_results:
            self.ess[time_index, 0] = optimization_results["ess"][-1]
        if "alpha" in optimization_results:
            self.alphas[time_index, 0] = optimization_results["alpha"][-1]
        if self.env.rewards.shape == self.rewards[time_index, ...].shape:
            written_idx = ~np.isnan(self.env.rewards)
            self.rewards[time_index, written_idx] = self.env.rewards[written_idx].copy()

    def warm_start(self, state: Tuple, time_index: int, n_iters: int):
        """Initial optimization before MPC.

        Returns:
            Mean optimization cost
            Standard deviation of the cost during optimization
            Trace result from optimization
        """
        res = self.optimize(state, time_index, n_iters, use_tqdm=True)

        if self.env.rewards.shape == self.rewards_warmstart.shape:
            written_idx = ~np.isnan(self.env.rewards)
            self.rewards_warmstart[written_idx] = self.env.rewards[written_idx].copy()

        return np.asarray(res["mean"]), np.asarray(res["std"]), res
