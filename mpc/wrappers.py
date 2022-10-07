"""
A wrapper around MuJoCo gym environments to control the true system.
Borrows several tricks from https://github.com/facebookresearch/mbrl-lib.
"""

import enum
import logging
from contextlib import ContextDecorator
from multiprocessing import Pool, Queue
from typing import Tuple, cast
from warnings import warn

import gym
import gym.wrappers
import numpy as np
from mujoco_py.builder import MujocoException

# used to communicate with MPC processes
read_q = Queue()
write_q = Queue()


# used to manage MPC processes
class Task(enum.Enum):
    rollout = 2
    close = 3


# required to reliably kill processes when using Keyboard interrupt
class KeyboardInterruptWorker(Exception):
    pass


class FreezeMujoco(object):
    """MuJoCo wrapper."""

    def __init__(self, env: gym.wrappers.TimeLimit):
        self._env = env
        self._init_state: np.ndarray = None
        self._elapsed_steps = 0
        self._step_count = 0

    def __enter__(self):
        self._init_state, self._elapsed_steps = MujocoEnvHandler.get_current_state(
            self._env
        )

    def __exit__(self, *_args):
        MujocoEnvHandler.set_env_state(
            (self._init_state, self._elapsed_steps), self._env,
        )


class MujocoEnvHandler(object):
    """Somewhat hacky getters and setters for variance MuJoCo-based environments."""

    freeze = FreezeMujoco

    @staticmethod
    def get_current_state(env: gym.wrappers.TimeLimit) -> Tuple:
        if hasattr(env.env, "get_env_state"):
            state = env.env.get_env_state()
        elif hasattr(env.env.sim, "get_state"):
            state = env.env.sim.get_state().flatten()
        elif hasattr(env.env, "data"):
            state = (
                env.env.data.qpos.ravel().copy(),
                env.env.data.qvel.ravel().copy(),
            )
        else:
            state = env.env._env.physics.get_state().copy()
        elapsed_steps = env._elapsed_steps
        return state, elapsed_steps

    @staticmethod
    def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit):
        if hasattr(env.env, "set_env_state"):
            env.set_env_state(state[0])
        elif hasattr(env.env.sim, "set_state"):
            env.env.sim.set_state_from_flattened(state[0])
        elif hasattr(env.env, "set_state"):
            env.set_state(*state[0])
        else:
            with env.env._env.physics.reset_context():
                env.env._env.physics.set_state(state[0])
        env._elapsed_steps = state[1]

    @classmethod
    def rollout_env(
        cls, env: gym.wrappers.TimeLimit, lookahead: int, action_sequence: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute a rollout of an action sequence, collect MDP result."""
        actions = []
        real_obses = []
        rewards = []
        with cls.freeze(cast(gym.wrappers.TimeLimit, env)):  # type: ignore
            try:
                for i in range(lookahead):
                    a = action_sequence[i]
                    next_obs, reward, done, _ = env.step(a)
                    actions.append(a)
                    real_obses.append(next_obs)
                    rewards.append(reward)
                    # don't use done flag? -- hard to interpret
                    # Reacher-v2 gets done after 50 timesteps
                    # if done:
                    #     rewards[-1] = np.nan
                    #     break

            except MujocoException as exc:
                rewards = [np.nan]
                real_obses = [np.nan]
                actions = [np.nan]
                a_max = np.max(np.abs(action_sequence))
                warn(f"MuJoCo rollout failed: Max|a| = {a_max}")
                logging.warning(f"MuJoCo rollout failed: Max|a| = {a_max}")

        return np.stack(real_obses), np.stack(rewards), np.stack(actions)


class ControlEnv(ContextDecorator):
    def __init__(self, env_name, dt, n_processes=1):
        self.env_name = env_name
        self.dt = dt
        self.state = None
        self.state_history = None
        self.n_processes = n_processes
        self.pool = Pool(n_processes, self.worker, (write_q, read_q, env_name))

    @staticmethod
    def worker(worker_read_q, worker_write_q, env_name):
        """Worker loop running in each proccess for efficient multiprocessing."""
        env = gym.make(env_name)
        while True:
            try:
                msg = worker_read_q.get(True)
                cmd, idx, data = msg
                if cmd == Task.rollout:
                    action_sequence, current_state = data
                    env = cast(gym.wrappers.TimeLimit, env)
                    env.reset()
                    MujocoEnvHandler.set_env_state(current_state, env)
                    obs, rewards, _ = MujocoEnvHandler.rollout_env(
                        env=env,
                        lookahead=action_sequence.shape[0],
                        action_sequence=action_sequence,
                    )
                    worker_write_q.put((idx, rewards))
                if cmd == Task.close:
                    break
            except KeyboardInterrupt:
                raise KeyboardInterruptWorker()

    def close(self):
        if self.pool is not None:
            for i in range(self.n_processes):
                write_q.put((Task.close, i, None))
            self.pool.close()
            self.pool.join()

    def __del__(self):
        self.close()

    def set_state(self, state):
        self.state = state

    def reset(self):
        self.state_history = None

    def __call__(self, action_sequences: np.ndarray) -> np.ndarray:
        """Send action samples to processes and return the returns."""
        assert self.state is not None
        n_samples, d_time, d_action = action_sequences.shape
        self.rewards = np.nan * np.ones((n_samples, d_time))

        try:
            args = [
                (Task.rollout, n, (action_sequences[n, ...], self.state))
                for n in range(n_samples)
            ]
            for arg in args:
                write_q.put(arg)
            out = [read_q.get() for _ in range(n_samples)]
            for (i, r) in out:
                self.rewards[i, : r.shape[0]] = r
            return -self.rewards.sum(1)
        except (Exception, KeyboardInterrupt):
            self.close()
            raise
