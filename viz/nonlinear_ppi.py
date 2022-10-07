import gif
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

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
gif.options.matplotlib["transparent"] = True

N = 1000
x_lim = 10
x_ = np.linspace(-x_lim, x_lim, N)
mu = 5
sigma2 = 2
mu_0 = 0
sigma_0 = 1
pdf_prior = stats.norm.pdf(x_, mu_0, sigma_0)

n_samples = 100
np.random.seed(0)
prior_samples = mu_0 + sigma_0 * np.random.randn(n_samples,)
prior_samples[0] = 2.91


def reward(x):
    ph = 6 * x
    return np.exp(-0.5 * (x - mu) ** 2 / sigma2) * np.abs(np.sin(ph)) - 1


r_ = reward(x_)


def _plot(callback):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax_f = ax.twinx()
    ax_iw = ax.twinx()
    ax.set_zorder(1)  # default zorder is 0 for ax1 and ax2
    ax.patch.set_visible(False)  # prevents ax1 from hiding ax2

    # right, left, top, bottom
    ax_iw.spines["right"].set_position(("outward", 70))
    ax_iw.xaxis.set_ticks([])

    ax.set_ylim(0, 6)
    ax_f.set_ylim(-1, 0)
    ax_iw.set_ylim(9e-3, 1)
    ax_iw.set_yscale("log")
    ax.set_xlim(-x_lim, x_lim)
    ax_f.plot(x_, r_, "k-", label="Reward, $R(x)$")
    ax.plot(x_, pdf_prior, "b", label="Prior, $p$")
    ax.fill_between(x_, pdf_prior, where=pdf_prior >= 0, color="b", alpha=0.2)
    callback(ax, ax_iw, prior_samples)
    ax.legend(loc="upper left")
    ax.set_ylabel("Probability density function")
    ax_f.set_ylabel("Reward, $R(x)$", color="k")
    ax_iw.set_ylabel("Importance weights, $q_\\alpha(x)$", color="m")
    ax_iw.tick_params(axis="y", colors="m")
    ax_iw.spines["right"].set_color("m")
    ax.set_xlabel("$x$")


# Decorate a plot function with @gif.frame (return not required):
@gif.frame
def plot(alpha):
    _plot(alpha)


def cem_callback(ax, ax_iw, prior_samples):
    ax.set_ylim(0, 3)
    r_ = reward(prior_samples)
    idxs = np.flip(np.argsort(r_))
    # CEM
    for j, elites in enumerate([50, 25, 10]):
        idx_elite = idxs[:elites]
        nw = np.zeros_like(prior_samples)
        nw[idx_elite] = 1 / elites
        for i, p in enumerate(prior_samples):
            ax_iw.vlines(p, 0, nw[i], color="r", alpha=0.1)

        mu = np.einsum("b,b->", nw, prior_samples)
        diff = prior_samples - mu
        sigma2 = np.einsum("b,b->", nw, diff ** 2)
        sigma = np.sqrt(sigma2)
        pdf_posterior = stats.norm.pdf(x_, mu, sigma)
        ax.plot(
            x_,
            pdf_posterior,
            "g",
            label="Next prior, $q_\\alpha\\rightarrow p$ (CEM)"
            if elites == 10
            else None,
        )
        ax.fill_between(
            x_, pdf_posterior, where=pdf_posterior >= 0, color="g", alpha=0.2
        )
        # print(alpha_, mu_0, sigma_0, mu, sigma, pdf_posterior.max())
        max_pdf = np.max(pdf_posterior)
        max_x = x_[np.argmax(pdf_posterior)]
        print(max_x, max_pdf)
        ax.annotate(
            f"$k={elites}$",
            color="g",
            xy=(max_x, max_pdf),
            xytext=(-3, 0.25 + 0.25 * j),
            # arrowprops=dict(facecolor='c', shrink=0.01)
            arrowprops=dict(edgecolor="g", arrowstyle="->"),
        )
    # ESSPS
    alpha_lower, alpha_upper = 1e-3, 1e3
    r_ = reward(prior_samples)
    r__ = (r_ - np.max(r_)) / np.abs(np.min(r_) - np.max(r_))
    print(r__.max(), r__.min())
    for j, ess in enumerate([2, 5, 10]):

        def ess_err(alpha, eps=1e-6):
            log_w_ = alpha * r__
            log_nw_ = log_w_ - logsumexp(log_w_)
            ess_ = np.exp(-logsumexp(2 * log_nw_))
            return np.abs(ess_ - ess)

        res = minimize_scalar(
            ess_err,
            method="brent",
            bounds=(alpha_lower, alpha_upper),
            options={"maxiter": 5000},
        )
        alpha = res.x
        log_w = alpha * r__
        nw = np.exp(log_w - logsumexp(log_w))
        nw = nw / nw.sum()
        ess_ = int(1 / np.sum(nw ** 2))
        for i, p in enumerate(prior_samples):
            ax_iw.vlines(p, 0, nw[i], color="m", alpha=0.1)
        mu_ = np.einsum("b,b->", nw, prior_samples)
        print(ess, ess_, alpha, mu_)
        diff = prior_samples - mu_
        sigma2 = np.einsum("b,b->", nw, diff ** 2)
        sigma = np.sqrt(sigma2)
        pdf_posterior = stats.norm.pdf(x_, mu_, sigma)
        ax.plot(
            x_,
            pdf_posterior,
            "c",
            label="Next prior, $q_\\alpha\\rightarrow p$ (ESSPS)" if j == 0 else None,
        )
        ax.fill_between(
            x_, pdf_posterior, where=pdf_posterior >= 0, color="c", alpha=0.2
        )
        max_pdf = np.max(pdf_posterior)
        max_x = x_[np.argmax(pdf_posterior)]
        print(max_x, max_pdf)
        ax.annotate(
            f"$\\hat{{N}}^*={ess}$",
            color="c",
            xy=(max_x, max_pdf),
            xytext=(-3, 0.25 + 0.25 * j),
            arrowprops=dict(edgecolor="c", arrowstyle="->"),
        )


# _plot(cem_callback)
# plt.savefig("ess_cem_nonlinear_ppi.png", bbox_inches="tight", dpi=300)


def lbps_callback(ax, ax_iw, prior_samples):
    ax.set_ylim(0, 3)
    # ESSPS
    alpha_lower, alpha_upper = 1e-3, 5e1
    r_ = reward(prior_samples)
    r__ = (r_ - np.max(r_)) / np.abs(np.min(r_) - np.max(r_))
    print(r__.max(), r__.min())
    _max = 1.0
    for j, delta in enumerate([0.6, 0.1, 0.5]):
        lambda_ = _max * np.sqrt((1 - delta) / delta)

        def lower_bound(alpha, eps=1e-6):
            log_w_ = alpha * r__
            log_nw_ = log_w_ - logsumexp(log_w_)
            nw_ = np.exp(log_nw_)
            ess = np.exp(-logsumexp(2 * log_nw_))
            ec = -np.einsum("b,b->", nw_, r__)
            err = lambda_ / np.sqrt(ess)
            return ec + err

        res = minimize_scalar(
            lower_bound,
            method="brent",
            bounds=(alpha_lower, alpha_upper),
            options={"maxiter": 5000},
        )
        alpha = np.clip(res.x, alpha_lower, alpha_upper)
        log_w = alpha * r__
        nw = np.exp(log_w - logsumexp(log_w))
        nw = nw / nw.sum()
        ess_ = int(1 / np.sum(nw ** 2))
        for i, p in enumerate(prior_samples):
            ax_iw.vlines(p, 0, nw[i], color="m", alpha=0.1)
        mu_ = np.einsum("b,b->", nw, prior_samples)
        diff = prior_samples - mu_
        sigma2 = np.einsum("b,b->", nw, diff ** 2) + 1e-2
        sigma = np.sqrt(sigma2)
        print(delta, mu_, sigma, ess_, alpha)
        pdf_posterior = stats.norm.pdf(x_, mu_, sigma)
        ax.plot(
            x_,
            pdf_posterior,
            "c",
            label="Next prior, $q_\\alpha\\rightarrow p$ (LBPS)" if j == 0 else None,
        )
        ax.fill_between(
            x_, pdf_posterior, where=pdf_posterior >= 0, color="c", alpha=0.2
        )
        max_pdf = np.max(pdf_posterior)
        max_x = x_[np.argmax(pdf_posterior)]
        ax.annotate(
            f"$\\delta={delta}$",
            color="c",
            xy=(max_x, max_pdf),
            xytext=(7.5, 2 - 0.35 * j),
            arrowprops=dict(edgecolor="c", arrowstyle="->"),
        )


_plot(lbps_callback)
plt.savefig("lbps_nonlinear_ppi.png", bbox_inches="tight", dpi=300)

exit()
# Build a bunch of "frames"
frames = []
alphas = np.exp(np.linspace(np.log(1e-3), np.log(100), 60))
alphas = np.concatenate((alphas, np.flip(alphas)))
for alpha_ in alphas:

    def callback(ax, ax_iw, prior_samples):
        log_w = alpha_ * reward(prior_samples)
        nw = np.exp(log_w - logsumexp(log_w))
        ess = int(1 / np.sum(nw ** 2))
        ax.set_title(f"$\\alpha$={alpha_:.2f}, ESS={ess:d}")
        for i, p in enumerate(prior_samples):
            ax_iw.vlines(p, 0, nw[i], color="m")

        mu_ = np.einsum("b,b->", nw, prior_samples)
        diff = prior_samples - mu_
        sigma2 = np.einsum("b,b->", nw, diff ** 2)
        sigma = np.sqrt(sigma2)
        pdf_posterior = stats.norm.pdf(x_, mu_, sigma)
        ax.plot(x_, pdf_posterior, "c", label="Next prior, $q_\\alpha\\rightarrow p$")
        ax.fill_between(
            x_, pdf_posterior, where=pdf_posterior >= 0, color="c", alpha=0.2
        )
        # print(alpha_, mu_0, sigma_0, mu, sigma, pdf_posterior.max())
        print(ess, mu_)

    frame = plot(callback)
    frames.append(frame)

# Specify the duration between frames (milliseconds) and save to file:
gif.save(frames, "iw_ppi.gif", duration=50)
