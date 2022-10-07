from os import makedirs
from os.path import exists
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from tqdm import trange


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(alg, n_epochs, n_steps, n_steps_test):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: " + alg.__name__)

    # MDP
    horizon = 1000
    gamma = 0.99
    env_name = "HumanoidStandup-v2"
    mdp = Gym(env_name, horizon, gamma)

    results_dir = Path(__file__).parent.resolve() / Path(env_name)
    print(results_dir)
    if not exists(results_dir):
        makedirs(results_dir)

    # Settings
    initial_replay_size = 64
    max_replay_size = int(1e6)
    batch_size = 256
    n_features = 256
    warmup_transitions = 5000
    tau = 0.005
    lr_alpha = 3e-4

    use_cuda = torch.cuda.is_available()

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
        use_cuda=use_cuda,
    )
    actor_sigma_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
        use_cuda=use_cuda,
    )

    actor_optimizer = {"class": optim.Adam, "params": {"lr": 3e-4}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(
        network=CriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": 3e-4}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
        use_cuda=use_cuda,
    )

    # Agent
    agent = alg(
        mdp.info,
        actor_mu_params,
        actor_sigma_params,
        actor_optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        warmup_transitions,
        tau,
        lr_alpha,
        critic_fit_params=None,
    )

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    s, *_ = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)

    logger.epoch_info(0, J=J, R=R, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        s, *_ = parse_dataset(dataset)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)

        logger.epoch_info(n + 1, J=J, R=R, entropy=E)

        if n % max(1, int(n_epochs * 0.05)) == 0 or n == n_epochs - 1:
            np.save(results_dir / "J.npy", np.array(J))
            np.save(results_dir / "R.npy", np.array(R))
            agent.save(results_dir / "model")
            states, actions, rewards, next_states, absorbing, last = parse_dataset(
                dataset
            )
            res = {"obs": states, "actions": actions, "rewards": rewards}
            np.savez(results_dir / "test_buffer.npz", **res)


if __name__ == "__main__":
    total_timesteps = 3e6
    steps_per_update = 5e3
    experiment(
        alg=SAC,
        n_epochs=int(total_timesteps // steps_per_update),
        n_steps=int(steps_per_update),
        n_steps_test=int(1e3),
    )
