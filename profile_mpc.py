import cProfile
import timeit

import gym
import numpy as np

import ppi.algorithms as algorithms
import ppi.policies as policies
import ppi.samplers as samplers
from mpc.mpc import Mpc
from mpc.wrappers import ControlEnv, MujocoEnvHandler
from run_mpc import get_control_timestep, get_state

# ENV = "door-v0"
ENV = "HumanoidStandup-v2"
TIMESTEPS = 250
HORIZON = 30
N_SAMPLES = 32
N_ITERS = 1
ANNEAL = 1.0
N_RUNS = 10


def benchmark(env, agent):
    state = get_state(env)
    for i in range(N_RUNS):
        _ = agent(state, i)


env = gym.make(ENV)
env.reset()
dt = get_control_timestep(env)
total_time_sequence = dt * np.arange(0, HORIZON)
limiter = policies.Limiter(lower=env.action_space.low, upper=env.action_space.high)
mean, covariance_in, covariance_out = policies.design_moments(
    env.action_space.low, env.action_space.high, ratio=1000.0
)

se_policy = policies.SquaredExponentialKernel(
    time_sequence=total_time_sequence,  # used for RBF construction
    action_dimension=env.action_space.shape[0],
    mean=mean,
    covariance_in=covariance_in,
    covariance_out=covariance_out,
    lengthscale=0.05,
    n_features=HORIZON,
    order=HORIZON // 2,
    sampler=samplers.MonteCarlo,
    limiter=limiter,
    use_derivatives=False,
)

wn_policy = policies.WhiteNoiseiid(
    time_sequence=total_time_sequence,  # used for RBF construction
    action_dimension=env.action_space.shape[0],
    mean=mean,
    covariance_in=covariance_in,
    covariance_out=covariance_out,
    sampler=samplers.MonteCarlo,
    limiter=limiter,
    use_derivatives=False,
)

cn_policy = policies.ColouredNoise(
    time_sequence=total_time_sequence,  # used for RBF construction
    action_dimension=env.action_space.shape[0],
    mean=mean,
    covariance_in=covariance_in,
    covariance_out=covariance_out,
    sampler=samplers.MonteCarlo,
    beta=2.0,
    limiter=limiter,
    use_derivatives=False,
)
icem = algorithms.iCem(n_elites=10, dimension=cn_policy.dim_features,)
env_mpc = ControlEnv(ENV, dt, 25)

mppi = algorithms.Mppi(alpha=10, dimension=cn_policy.dim_features,)

for n_samples in [16, 128, 1024]:
    icem_agent = Mpc(
        env_mpc, dt, icem, cn_policy, TIMESTEPS, HORIZON, n_samples, N_ITERS, ANNEAL,
    )
    lbps = algorithms.Lbps(delta=0.1, dimension=se_policy.dim_features,)
    ppi_agent = Mpc(
        env_mpc, dt, lbps, se_policy, TIMESTEPS, HORIZON, n_samples, N_ITERS, ANNEAL,
    )
    mppi_agent = Mpc(
        env_mpc, dt, mppi, se_policy, TIMESTEPS, HORIZON, n_samples, N_ITERS, ANNEAL,
    )

    mppi_wn_agent = Mpc(
        env_mpc, dt, mppi, wn_policy, TIMESTEPS, HORIZON, n_samples, N_ITERS, ANNEAL,
    )

    runs = 5
    # ppi_time = timeit.timeit(lambda: benchmark(env, ppi_agent), number=runs) / N_RUNS / runs
    # icem_time = timeit.timeit(lambda: benchmark(env, icem_agent), number=runs) / N_RUNS / runs
    # mppi_time = timeit.timeit(lambda: benchmark(env, mppi_agent), number=runs) / N_RUNS / runs
    mppi_wn_time = (
        timeit.timeit(lambda: benchmark(env, mppi_wn_agent), number=runs)
        / N_RUNS
        / runs
    )
    # print(f"{n_samples} iCEM {icem_time:.2f}")
    # print(f"{n_samples} ppi {ppi_time:.2f}")
    # print(f"{n_samples} mppi {mppi_time:.2f}")
    print(f"{n_samples} mppi {mppi_wn_time:.2f}")
