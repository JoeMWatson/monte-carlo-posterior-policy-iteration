"""
Environments for policy search experiments.

Builds off code written with Johannes Silberbauer and Michael Lutter.
Also builds of experiment design of Pascal Klink.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from ball_in_a_cup import BallInCupParallelTrajectoryEvaluator
from ball_in_a_cup import BallInCupSim as BallInCupSim
from ball_in_a_cup import BicType, evaluate_trajectory
from tqdm import tqdm

from utils import NullContext, VideoRenderStream

__all__ = ["BallInACup", "Test"]


class Base(ABC):

    dim_action: int
    success_rate = []

    @abstractmethod
    def map_actions_to_joints(
        self, action_sequences: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def batch_rollout(self, q: np.ndarray, qd: np.ndarray) -> (np.ndarray, np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def episodic_cost(self, traces: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Returns: episodic reward per sample and success rate
        """
        raise NotImplementedError

    def __call__(self, action_sequences: np.ndarray) -> np.ndarray:
        # run policy
        qs, qds = self.map_actions_to_joints(action_sequences)
        traces, trajectories = self.batch_rollout(qs, qds)
        # reward
        costs, success_flags = self.episodic_cost(traces)
        self.success_rate += [(1.0 * success_flags).mean()]
        return costs


class Test(Base):

    dim_action = 2
    dim_dof = 2
    t = np.linspace(0, 1, 100)
    action_0 = np.zeros((2,))
    condition = False

    def __init__(self):
        amps = np.linspace(-1, 1, self.dim_action)
        self.qs_g = np.concatenate(
            [
                amps[k] * np.sin(4 * (k + 1) * self.t)[:, None] / 2
                for k in range(self.dim_action)
            ],
            axis=1,
        )

    def map_actions_to_joints(self, action_sequence):
        return (
            action_sequence[..., : self.dim_dof],
            action_sequence[..., self.dim_dof :],
        )

    def batch_rollout(self, qs, qds):
        return qs, qs

    def episodic_cost(self, traces):
        qs = traces
        n_, t_, a_ = traces.shape
        sq_err = np.power(qs - self.qs_g[None, :, :], 2)
        costs = np.einsum("bij->b", sq_err) / (t_ * a_)

        fig, axs = plt.subplots(qs.shape[-1])
        for i_, ax in enumerate(axs):
            ax.plot(qs[:, :, i_].T, alpha=0.3)
            ax.plot(self.qs_g[:, i_], "k-")

        return costs, np.zeros_like(costs)


class BallInACup(Base):

    dim_action = 2
    dim_dof = 4
    time_horizon = 2

    action_0 = np.array([0.0, 1.5707])
    condition = True

    starting_state = np.array([np.pi / 2, np.pi * 1 / 4, 0.0, np.pi * 1 / 4])
    action_indices = np.array([1, 3])

    def __init__(self):
        self.dt = BallInCupSim().effective_dt
        self.t = np.linspace(0, self.time_horizon, int(self.time_horizon / self.dt))

    def map_joints_to_actions(self, joints: np.ndarray) -> np.ndarray:
        return joints[..., self.action_indices]

    def map_actions_to_joints(
        self, action_sequences: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        # if the action sequence only encodes two joint trajectories we map those to the
        # first and last joint of the arm
        dim_samp, dim_time, dim_state = action_sequences.shape
        assert dim_state == self.dim_action * 2
        qs_ = np.zeros((dim_samp, dim_time, self.dim_dof))
        qds_ = np.zeros((dim_samp, dim_time, self.dim_dof))
        qs, qds = (
            action_sequences[..., : self.dim_action],
            action_sequences[..., self.dim_action :],
        )
        qs_[..., self.action_indices] = qs
        qds_[..., self.action_indices] = qds
        qs, qds = qs_, qds_
        return qs, qds

    def callback(
        self,
        iteration: int,
        f: Callable,
        actions: np.ndarray,
        costs: np.ndarray,
        policy,
        path: Path,
    ):
        if path is None:
            return
        else:
            q, qd = self.map_actions_to_joints(actions)
            assert q.shape[0] == qd.shape[0]
            n_samp = q.shape[0]
            q0 = np.array([0.0, 0.0, 0.0, 1.5707])
            for n in tqdm(range(n_samp)):
                # null = NullContext
                video_render_ctx = VideoRenderStream(
                    path / f"ball-in-cup_{iteration}_{n}.mp4",
                    Path(__file__).parent.resolve(),
                )
                with video_render_ctx as video_stream:
                    exe_params = dict(
                        # stabilize_current_pos=True, verbose=True, video_writer=video_stream
                        stabilize_current_pos=True,
                        verbose=False,
                        video_writer=video_stream,
                    )
                    evaluate_trajectory(
                        q0=q0,
                        trj=(q[n, ...], qd[n, ...]),
                        trj_kwargs=exe_params,
                        sim_init_kwargs=dict(type_=BicType.cylinder),
                    )

    def batch_rollout(self, q: np.ndarray, qd: np.ndarray) -> (np.ndarray, np.ndarray):
        assert q.shape[0] == qd.shape[0]
        n_samp = q.shape[0]
        q0 = np.array([0.0, 0.0, 0.0, 1.5707])
        evaluator = BallInCupParallelTrajectoryEvaluator(q0)
        trajectories = [(q[i, ...], qd[i, ...]) for i in range(n_samp)]
        video_render_ctx = NullContext()
        with video_render_ctx as video_stream:
            exe_params = dict(
                stabilize_current_pos=True, verbose=False, video_writer=video_stream,
            )
            traces = evaluator.eval(
                trajectories, exe_params, dict(type_=BicType.cylinder)
            )
        return traces, traces

    def episodic_cost(self, traces: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Returns: episodic reward per sample
        """
        from ball_in_a_cup import BallInCupRewardParams, compute_reward

        reward_params = BallInCupRewardParams(
            state_reward_type="dipole_potential",
            joint_velocity_penalty_factor=3e-2,
            joint_position_penalty_factor=7.5e-2,
            ball_velocity_penalty_factor=0.0,
            cup_inner_radius=0.069 / 2.0,
            reward_dipole_eps=1e-3,
            reward_dipole_beta=1e-1,
            reward_min_weight=0.5,
        )
        rewards, successes = map(
            np.asarray, zip(*[compute_reward(trace, reward_params) for trace in traces])
        )
        rewards -= 100.0
        pc = successes.sum() / successes.shape[0]
        print(f"Mean return: {-rewards.mean():.2f}, std. dev return: {rewards.std():.2f}, success rate: {pc:.2f}")
        return -rewards, successes
