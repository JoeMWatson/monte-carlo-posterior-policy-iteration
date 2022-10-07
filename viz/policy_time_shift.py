import gif
import matplotlib.pyplot as plt
import numpy as np

import ppi.policies as policies
from ppi.algorithms import Ais, Cem, Lbps
from ppi.samplers import MonteCarlo

plt.rcParams.update(
    {
        "figure.autolayout": True,
        "font.size": 15,
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern Roman"],
    }
)

gif.options.matplotlib["dpi"] = 300

N_SAMPLES = 1000
N_ITERS = 50
N_EVAL_SAMPLES = 10

np.random.seed(0)
n_ = 30
tt_ = 1
dt = tt_ / n_
n_long = int(1.1 * n_)
n_res = 1000
t_ = dt * np.arange(0, n_)
t_long = dt * np.arange(0, 2 * n_)
# t_long = dt * np.arange(0, n_ + 1)
# t_res = np.linspace(0, tt_, n_res)
# t_long_res = np.linspace(0, tt_long, n_res)

# Task: quadratic periodic trajectory
def u_d(tau):
    return 1.0 * (np.cos(2 * np.pi * tau) > 0.0)  # - 0.5


u_opt = u_d(t_)[:, None]


def cost(u):
    err = np.abs(u - u_opt)
    return np.sum(err, axis=1)[:, 0]


def cost_step(t, u):
    return np.abs(1 - u_d(t)) * np.abs(u - u_d(t))


max_, min_ = np.atleast_1d(u_opt.max()), np.atleast_1d(u_opt.min())
mean_ = (max_ + min_) / 2.0
print(max_, min_, mean_)
limiter = policies.Limiter(upper=max_, lower=min_)

POLICIES = {
    # "rbf": ("Radial basis function features", policies.RbfFeatures),
    # "se": ("Squared exponential kernel", policies.SquaredExponentialKernel),
    # "rff": ("Quadrature random Fourier features", policies.RffFeatures),
    "wn": ("White noise kernel", policies.WhiteNoiseIid),
}

for name, data in POLICIES.items():
    print(name)
    title, policy_class = data
    policy = policy_class(
        lengthscale=0.2,
        time_sequence=t_long if name == "rbf" else t_,
        action_dimension=1,
        mean=mean_,
        covariance_in=np.array([1e2]),
        covariance_out=0.5 * np.array([[1e-2]]),
        # lengthscale=1.0,
        # period=2 * np.pi,
        n_features=50,
        order=50,
        sampler=MonteCarlo,
        limiter=limiter,
        use_derivatives=False,
    )
    policy.compute_prior(t_)
    solver = Cem(
        alpha=N_SAMPLES // 10,
        n_elites=N_SAMPLES // 10,
        n_elites_pc=0.1,
        delta=0.01,
        base_entropy=-200,
        entropy_rate=0.99,
        dimension=policy.dim_features,
    )
    res = solver(cost, policy, N_SAMPLES, N_ITERS)
    mean, *_ = policy.predict()

    def _plot(time_sequence):
        print(time_sequence.shape)
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.set_title(title)
        ax.set_xlim(-dt, 2 * dt * n_)
        policy.update_timesteps(time_sequence, 1.0)
        policy.compute_prior(time_sequence)
        samples, _ = policy(N_EVAL_SAMPLES)
        ax.plot(t_long, u_d(t_long), "k--")
        ax.plot(time_sequence, samples[:, :, 0].T, "c-", alpha=0.5)
        ax.plot(t_, mean, "b.-")
        ax.set_xticks([t_.min(), t_.max(), time_sequence.min(), time_sequence.max()])
        ax.set_xticklabels(["$t_1$", "$t_2$", "$t_3$", "$t_4$"])
        ax.set_ylabel("$a$")

    @gif.frame
    def plot(time_sequence):
        _plot(time_sequence)

    time_sequences = [dt * np.arange(2, n_ + i) for i in range(2, n_, 1)]
    time_sequences += list(reversed(time_sequences))
    # _plot(time_sequences[0])
    # plt.show()
    # exit()
    frames = []
    for time_sequence_ in time_sequences:
        frame = plot(time_sequence_)
        frames.append(frame)

    # Specify the duration between frames (milliseconds) and save to file:
    gif.save(frames, f"{name}_policy_timeshift.gif", duration=50)
