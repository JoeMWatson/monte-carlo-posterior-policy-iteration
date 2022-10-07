import gif
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

plt.rcParams.update(
    {
        "font.size": 15,
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern Roman"],
    }
)

gif.options.matplotlib["dpi"] = 300
gif.options.matplotlib["transparent"] = True
gif.options.matplotlib["bbox_inches"] = "tight"

N = 1000
x_lim = 10
x_ = np.linspace(-x_lim, x_lim, N)
mu = 5
sigma2 = 0.5
mu_0 = -5
sigma_0 = 1
pdf_prior = stats.norm.pdf(x_, mu_0, sigma_0)


def quad(x):
    return 0.5 * (x - mu) ** 2 / sigma2


def dquad(x):
    return (x - mu) / sigma2


def d2quad(x):
    return np.atleast_1d(1 / sigma2)


f_ = quad(x_)

# Decorate a plot function with @gif.frame (return not required):
@gif.frame
def plot(alpha):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(0, 6)
    ax.set_title(f"$\\alpha$={alpha:.2f}")
    ax_f = ax.twinx()
    ax_f.plot(x_, -f_, "k-")
    ax.plot(x_, pdf_prior, "b", label="Prior, $p$")
    ax.fill_between(x_, pdf_prior, where=pdf_prior >= 0, color="b", alpha=0.2)
    sigma2 = 1 / (sigma_0 ** 2 + alpha * d2quad(mu_0))
    mu = mu_0 - alpha * sigma2 * dquad(mu_0)
    sigma = np.sqrt(sigma2)
    pdf_posterior = stats.norm.pdf(x_, mu, sigma)
    ax.plot(x_, pdf_posterior, "c", label="Posterior, $q_\\alpha$")
    ax.fill_between(x_, pdf_posterior, where=pdf_posterior >= 0, color="c", alpha=0.2)
    print(alpha, mu_0, sigma_0, mu, sigma, pdf_posterior.max())
    ax.legend(loc="upper left")
    ax.set_ylabel("Probability density function")
    ax_f.set_ylabel("$R(x) = -0.5 * (x-5)^2$")
    ax.set_xlabel("$x$")


if __name__ == "__main__":
    # Build a bunch of "frames"
    frames = []
    alphas = np.exp(np.linspace(np.log(1e-3), np.log(100), 30))
    alphas = np.concatenate((alphas, np.flip(alphas)))
    for alpha_ in alphas:
        frame = plot(alpha_)
        frames.append(frame)

    # Specify the duration between frames (milliseconds) and save to file:
    gif.save(frames, "gaussian_ppi.gif", duration=50)
