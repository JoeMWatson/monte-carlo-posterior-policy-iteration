"""
Priors used for posterior policy iteration methods.
Includes white noise, kernels and finite feature approximations.
"""

import logging
from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple

import colorednoise
import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import gamma, kv, logsumexp
from scipy.stats import multivariate_normal
from tqdm import tqdm

import ppi.samplers as samplers

__all__ = [
    "RbfFeatures",
    "RffFeatures",
    "SquaredExponentialKernel",
    "WhiteNoiseKernel",
    "WhiteNoiseIid",
    "ColouredNoise",
    "SmoothActionNoise",
    "SmoothExplorationNoise",
    "Matern12Kernel",
    "Matern32Kernel",
    "Matern52Kernel",
    "PeriodicKernel",
    "LinearGaussianDynamicalSystemKernel",
]

EPS = np.finfo(np.float64).tiny
SIGMA_MIN = 1e-6


def design_moments(
    upper: np.ndarray, lower: np.ndarray, ratio: float
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Takes actuator limits and computes a suitable matrix normal distribution that explores this range.
    """
    mean = (upper + lower) / 2
    action_variance = (upper - lower) ** 2 / 4
    covariance_out = action_variance / ratio
    covariance_in = np.array([ratio])
    return mean, covariance_in, np.diag(covariance_out)


def symmetric(mat: np.ndarray):
    """Ensure (estimated) covariance matrix is symmetric."""
    assert len(mat.shape) == 2
    return 0.5 * (mat + mat.T)


def factorized(mat: np.ndarray):
    """Convert a full covariance into a factorized one (i.e. off-diagonals are zero)."""
    assert len(mat.shape) == 2
    return np.diag(np.diag(mat))


class Limiter(object):
    """Clip actions within actuator range."""

    def __init__(self, upper: np.ndarray, lower: np.ndarray):
        assert upper.shape == lower.shape
        self.dim = upper.shape[0]
        self.upper, self.lower = upper, lower

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[-1] == self.dim, f"{x.shape}, {self.dim}"
        return np.clip(x, a_max=self.upper, a_min=self.lower).reshape(x.shape)


def m_projection(
    log_w: np.ndarray, samples: np.ndarray
) -> (np.ndarray, np.ndarray, float):
    """
    Perform weighted M-projection (i.e. moment match) on samples, returning a multivariate normal.
    """
    lse_w = logsumexp(log_w)
    log_nw = log_w - lse_w
    nw = np.exp(log_nw)
    ess = np.exp(-logsumexp(2 * log_nw))
    mu = np.einsum("b, bi->i", nw, samples)
    dist = samples - mu[None, :]
    # biased ML estimate using the ESS.
    sigma = np.einsum("b,bi,bj->ij", nw, dist, dist)
    sigma = 0.5 * (sigma + sigma.T)
    return mu, sigma, ess


def m_projection_mavn(
    log_w: np.ndarray,
    samples: np.ndarray,
    covariance_in: np.ndarray,
    covariance_out: np.ndarray,
    iterations=1,
    update_out=False,
    use_tqdm=False,
) -> (np.ndarray, np.ndarray, np.ndarray, float):
    """
    Perform weighted M-projection (i.e. moment match) on samples, returning a matrix normal.
    As the matrix normal MLE is iterative, it requires initial guesses and can be iterated.
    log_w: log un-normalized weights
    U' = \sum_n (X_n - M) V\inv (X_n-M)^T
    V' = \sum_n (X_n - M)^T U\inv (X_n-M)
    """
    assert iterations > 0
    d_in = covariance_in.shape[0]
    d_out = covariance_out.shape[0]
    scale = np.power(np.linalg.det(covariance_in), 1.0 / d_in)
    in_reg = 1e-1 * scale * np.eye(d_in)
    lse_w = logsumexp(log_w)
    log_nw = log_w - lse_w
    nw = np.exp(log_nw)
    ws = np.exp(lse_w)
    ess = np.exp(-logsumexp(2 * log_nw))
    mu = np.einsum("b, bij->ij", nw, samples)
    diff = samples - mu[None, ...]
    iters_ = range(iterations)
    iters = tqdm(iters_) if use_tqdm else iters_
    for _ in iters:
        # input covariance
        # should be factorized
        out_inv = np.diag(1.0 / np.diag(covariance_out))
        # unbiased ML estimate using the ESS.
        covariance_in = np.einsum("b,bij,jk,blk->il", nw, diff, out_inv, diff) / d_out
        covariance_in = symmetric(covariance_in)
        if update_out:
            # factorize
            in_inv = np.diag(1.0 / np.diag(covariance_in))
            # unbiased ML estimate using the ESS.
            # covariance_out = np.einsum("b,bij,ik,bkl->jl", nw, diff, in_inv, diff) * (ess / (ess-1)) / d_in
            covariance_out = (
                np.einsum("b,bij,ik,bkl->jl", nw, diff, in_inv, diff) / d_in
            )
            covariance_out = symmetric(covariance_out)
            # covariance_out = np.diag(np.diag(covariance_out))  # keep factorized
    return mu, covariance_in, covariance_out, ess


def multivariate_gaussian_kl(
    mu_1: np.ndarray, sigma_1: np.ndarray, mu_2: np.ndarray, sigma_2: np.ndarray
) -> float:
    """Compute Kullback Leibler diveragnce between two multivariate Gaussians."""
    d = sigma_1.shape[0]
    diff = mu_2 - mu_1
    return 0.5 * (
        np.log(max(EPS, np.linalg.det(sigma_2)))
        - np.log(max(EPS, np.linalg.det(sigma_1)))
        + np.trace(np.linalg.solve(sigma_2, sigma_1))
        + diff @ np.linalg.solve(sigma_2, diff)
        - d
    )


def vec(x: np.ndarray) -> np.ndarray:
    """
    Turn matrix into vector.
    https://stackoverflow.com/questions/25248290/most-elegant-implementation-of-matlabs-vec-function-in-numpy
    """
    shape = x.shape
    if len(shape) == 3:
        a, b, c = shape
        return x.reshape((a, b * c), order="F")
    else:
        return x.reshape((-1, 1), order="F")


def matrix_gaussian_kl(
    mean_1: np.ndarray,
    cov_in_1: np.ndarray,
    cov_out_1: np.ndarray,
    mean_2: np.ndarray,
    cov_in_2: np.ndarray,
    cov_out_2: np.ndarray,
) -> float:
    """Compute Kullback Leibler diveragnce between two matrx normal Gaussians.
    ref: https://statproofbook.github.io/P/matn-kl
    """
    n, p = mean_1.shape
    diff = mean_2 - mean_1
    sf1 = p / np.trace(cov_out_1)
    sf2 = p / np.trace(cov_out_2)
    cov_out_1 = cov_out_1 * sf1
    cov_out_2 = cov_out_2 * sf2
    cov_in_1 = cov_in_1 / sf1
    cov_in_2 = cov_in_2 / sf2
    return (
        0.5
        * (
            n * np.log(max(EPS, np.linalg.det(cov_out_2)))
            - n * np.log(max(EPS, np.linalg.det(cov_out_1)))
            + p * np.log(max(EPS, np.linalg.det(cov_in_2)))
            - p * np.log(max(EPS, np.linalg.det(cov_in_1)))
            + np.trace(
                np.kron(
                    np.linalg.solve(cov_out_2, cov_out_1),
                    np.linalg.solve(cov_in_2, cov_in_1),
                )
            )
            + vec(diff).T
            @ vec(np.linalg.solve(cov_in_2, np.linalg.solve(cov_out_2, diff.T).T))
            - n * p
        ).item()
    )


def multivariate_gaussian_entropy(sigma: np.ndarray, d: int) -> float:
    return 0.5 * np.log(max(EPS, np.linalg.det(sigma))) + (d / 2) * (
        1 + np.log(2 * np.pi)
    )


def matrix_normal_entropy(
    covariance_in: np.ndarray, covariance_out: np.ndarray, d_in: int, d_out: int
) -> float:
    """The Matrix normal covariance are invariant to a scale factor.
    Due to numerical issues, relating to the high dimension and low values of
    covariance_out, we scale the covariances to normalize the output but preserving
    the true entropy.
    """
    sf = d_out / np.trace(covariance_out)
    logdet_in = d_out * np.log(max(EPS, np.linalg.det(covariance_in / sf)))
    logdet_out = d_in * np.log(max(EPS, np.linalg.det(sf * covariance_out)))
    ent = 0.5 * (logdet_in + logdet_out) + (d_in * d_out / 2) * (1 + np.log(2 * np.pi))
    return ent


class GaussianPolicy(object):

    name = "Gaussian"

    def __init__(self, mu: np.ndarray, sigma: np.ndarray, sampler, diagonal=False):
        self.sigma_init = sigma.copy()
        self.mu, self.sigma, self.sampler = mu, sigma, sampler
        self.diagonal = diagonal  # factorized covariance

    @property
    def entropy(self):
        return multivariate_gaussian_entropy(self.sigma, self.sigma.shape[0])

    def __call__(self, n_samples: np.ndarray):
        samples = self.sampler(self.mu, self.sigma, n_samples, standard_gaussian=False)
        return samples, samples

    def weighted_update(
        self, log_weights: np.ndarray, samples: np.ndarray, update_covariance_in=True
    ):
        mu, sigma = self.mu.copy(), self.sigma.copy()
        mu_, sigma_, ess = m_projection(log_weights, samples)
        if self.diagonal:
            sigma_ = factorized(sigma_)
        self.mu = mu_
        try:
            if update_covariance_in:
                np.linalg.cholesky(sigma_)
                self.sigma = sigma_
        except np.linalg.LinAlgError:
            logging.warning(
                "Fitted multivariate normal not positive definite. Regularizing covariance."
            )
            # typically means the variance has got really small
            self.sigma += SIGMA_MIN * np.eye(self.mu.shape[0])
        return ess, multivariate_gaussian_kl(self.mu, self.sigma, mu, sigma)

    @staticmethod
    def smooth(x, y, alpha):
        return alpha * x + (1 - alpha) * y

    def smooth_update(self, mu, sigma, alpha):
        self.mu, self.sigma = map(
            self.smooth, [mu, sigma], [self.mu, self.sigma], [alpha, alpha]
        )

    def reset_covariance(self):
        self.sigma = self.sigma_init.copy()


def null_limiter(action_sequence):
    return action_sequence


class BasePrimitive(ABC):
    """Primitive captures kernel- and feature-based policies, that are matrix valued."""

    dim_features: int
    dim_out: int
    t: np.ndarray
    map_sequence: np.ndarray

    def __init__(
        self,
        time_sequence: np.ndarray,
        action_dimension: np.ndarray,
        mean: np.ndarray,
        covariance_in: np.ndarray,
        covariance_out: np.ndarray,
        sampler,
        limiter=null_limiter,
        use_derivatives=False,
    ):
        self.t = time_sequence
        self.mean = mean
        self.dim_out = action_dimension
        self.covariance_in_init = covariance_in.copy()
        self.covariance_in = covariance_in
        self.covariance_in_sqrt = np.linalg.cholesky(covariance_in)
        self.covariance_out = covariance_out
        self.cov_out_sqrt = np.linalg.cholesky(self.covariance_out)
        self._sampler = sampler
        self.limiter = limiter
        self.use_derivatives = use_derivatives

    def reset_covariance(self):
        self.covariance_in = self.covariance_in_init.copy()
        self.covariance_in_sqrt = np.linalg.cholesky(self.covariance_in)

    @property
    def sampler(self):
        return self._sampler(self.dim_sample)

    @property
    def dim_sample(self):
        return self.dim_features * self.dim_out

    @property
    def mu_z(self):
        return np.zeros((self.dim_sample,))

    @property
    def sigma_z(self):
        return np.eye(self.dim_sample)

    @abstractmethod
    def __call__(self, n_samples: int) -> (np.ndarray, np.ndarray):
        raise NotImplementedError

    def condition(self, t, action):
        dim_t = t.shape[0]
        assert action.shape == (
            dim_t,
            self.dim_out,
        ), f"{action.shape} ~= {dim_t, self.dim_out}"
        self._condition(t, action)

    def _condition(self, t, action):
        raise NotImplementedError

    def update(self, mean: np.ndarray, covariance_in: np.ndarray):
        assert mean.shape == (self.dim_features, self.dim_action)
        assert covariance_in.shape == (self.dim_features, self.dim_features)
        self.mean = mean
        self.covariance_in = covariance_in
        self.covariance_in_sqrt = np.linalg.cholesky(covariance_in)

    def base_sample(self, n_samples: int):
        # sample from N(0, I)
        return self.sampler(
            self.mu_z, self.sigma_z, n_samples, standard_gaussian=True
        ).reshape((-1, self.dim_features, self.dim_out))

    @staticmethod
    def smooth(x, y, alpha):
        return alpha * x + (1 - alpha) * y

    def smooth_update(self, mean, covariance_in, alpha):
        self.mean, self.covariance_in = map(
            self.smooth,
            [mean, covariance_in],
            [self.mean, self.covariance_in],
            [alpha, alpha],
        )

    @property
    def entropy(self):
        """
        TODO: configure computation only when analysisng performance to save computation.
        """
        return 0.0
        # return matrix_normal_entropy(
        #     self.covariance_in, self.covariance_out, self.dim_features, self.dim_out
        # )

    def compute_prior(self, plan_time):
        self.t = plan_time


class BaseFeatures(BasePrimitive):
    """Feature-based policies."""

    def update_timesteps(self, t, anneal=1.0, eps=1e-5):
        self.t = t
        if anneal < 1.0:
            self.covariance_in = (
                anneal * self.covariance_in + (1 - anneal) * self.covariance_in_init
            )
            self.covariance_in_sqrt = np.linalg.cholesky(self.covariance_in)

    def __call__(self, n_samples: int) -> (np.ndarray, np.ndarray):
        """Return n_samples for policy, giving actions and weights."""
        feat_t = self.feat(self.t)
        zs = self.base_sample(n_samples)
        ws = self.mean[None, :, :] + np.einsum(
            "bij, ki, jl->bkl", zs, self.covariance_in_sqrt, self.cov_out_sqrt.T
        )
        xs = self.mean_fn[None, None, :] + np.einsum("bij,ki->bkj", ws, feat_t)
        if self.use_derivatives:
            dfeat_t = self.dfeat(self.t)
            dxs = np.einsum("bij,ki->bkj", ws, dfeat_t)
            ys = np.concatenate((xs, dxs), axis=-1)
        else:
            ys = xs
        if self.limiter:
            return self.limiter(ys), ws
        else:
            return ys, ws

    def predict(
        self, only_mean=False
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """Return mean and covariances for timesteps."""
        feat_t = self.feat(self.t)
        mu_y = self.limiter(self.mean_fn[None, :] + feat_t @ self.mean)
        if only_mean:
            return mu_y
        else:
            sigma_y_in = feat_t @ self.covariance_in @ feat_t.T
            sigma_y_out = self.covariance_out
            std_y_out = np.sqrt(
                np.einsum("b,c->bc", np.diag(sigma_y_in), np.diag(sigma_y_out))
            )
            return mu_y, sigma_y_in, sigma_y_out, std_y_out

    @abstractmethod
    def feat(self, t: float) -> np.ndarray:
        """Primitive features"""
        raise NotImplementedError

    @abstractmethod
    def dfeat(self, t: float) -> np.ndarray:
        """Primitive derivative features w.r.t. time."""
        raise NotImplementedError

    def _condition(self, t: float, action: np.ndarray):
        """Condition GP to an action at a certain time t."""
        f = self.feat(t)
        cov_0 = self.covariance_in
        # Minka equations
        S_xx = f.T @ f + np.linalg.inv(cov_0)
        S_yx = (action - self.mean_fn[None, :]).T @ f + self.mean.T @ np.linalg.inv(
            cov_0
        )
        self.mean = np.linalg.solve(S_xx, S_yx.T)
        self.covariance_in = np.linalg.inv(S_xx)

    def weighted_update(
        self, log_weights: np.ndarray, samples: np.ndarray, update_covariance_in=True
    ) -> (float, float):
        """Importance weighted maximum likelihood fit.

        Returns:
            Effective sample size of weights
            KL divergence of update
        """
        mean, covariance_in, covariance_in_sqrt, covariance_out = (
            self.mean.copy(),
            self.covariance_in.copy(),
            self.covariance_in_sqrt.copy(),
            self.covariance_out.copy(),
        )
        mean_, covariance_in_, covariance_out_, ess = m_projection_mavn(
            log_weights, samples, self.covariance_in, self.covariance_out
        )
        try:
            covariance_in_ += 1e-12 * np.eye(self.dim_features)
            covariance_in_sqrt_ = np.linalg.cholesky(covariance_in_)
            self.mean = mean_
            if update_covariance_in:
                self.covariance_in, self.covariance_in_sqrt = (
                    covariance_in_,
                    covariance_in_sqrt_,
                )
            kl = 0.0
            # save computation for experiments
            # kl = matrix_gaussian_kl(
            #     self.mean,
            #     self.covariance_in,
            #     self.covariance_out,
            #     mean,
            #     covariance_in,
            #     self.covariance_out,
            # )

        except np.linalg.LinAlgError:
            logging.warning(f"Update error, ESS={ess}")
            logging.warning(f"weights: [{log_weights.max()}, {log_weights.min()}]")
            eigvals = np.linalg.eigvalsh(covariance_in_)
            logging.warning(f"Eigvals: [{eigvals.max()}, {eigvals.min()}]")
            diags = np.diag(covariance_in_)
            logging.warning(f"Diagonals: [{diags.max()}, {diags.min()}]")
            logging.exception("Weighted update error, reverting")
            kl = 0.0
            ess = samples.shape[0]
            moments = (mean, covariance_in, covariance_in_sqrt, covariance_out_)
            (
                self.mean,
                self.covariance_in,
                self.covariance_in_sqrt,
                self.covariance_out,
            ) = moments
        return ess, kl


class RbfFeatures(BaseFeatures):
    """Radial basis function features."""

    def __init__(
        self,
        time_sequence: np.ndarray,
        action_dimension: int,
        mean: np.ndarray,
        covariance_in: np.ndarray,
        covariance_out: np.ndarray,
        lengthscale: float,
        n_features: int,
        sampler,
        use_derivatives,
        limiter=null_limiter,
        add_bias=False,
        *args,
        **kwargs,
    ):
        self.dim_features = time_sequence.shape[0]
        assert lengthscale > 0.0
        assert mean.shape == (
            action_dimension,
        ), f"{mean.shape} != {(action_dimension,)}"
        assert covariance_in.shape == (1,)
        self.add_bias = add_bias
        self.ls = lengthscale / np.sqrt(2)
        self.n_features = n_features
        self.dim_features = n_features
        if self.add_bias:
            self.dim_features += 1

        t_min, t_max = time_sequence[0], time_sequence[-1]
        self.centres = np.linspace(t_min, t_max, n_features)[:, None]
        self.norm = 1 / np.sqrt(np.sqrt(np.pi) * n_features * self.ls)
        mean_ = np.zeros((self.dim_features, action_dimension))
        self.mean_fn = mean
        covariance_in_ = covariance_in.item() * np.eye(self.dim_features)

        super().__init__(
            time_sequence,
            action_dimension,
            mean_,
            covariance_in_,
            covariance_out,
            sampler,
            limiter,
            use_derivatives,
        )

    def feat(self, t):
        f = (
            self.norm
            * np.exp(-0.5 * np.power((t[:, None] - self.centres.T) / self.ls, 2)),
        )
        if self.add_bias:
            f += (np.ones_like(t)[:, None],)
        f = np.concatenate(f, axis=1)
        return f

    def dfeat(self, t):
        diff = t[:, None] - self.centres.T
        f = (
            -self.norm
            * diff
            / np.power(self.ls, 2)
            * np.exp(-0.5 * np.power(diff / self.ls, 2)),
        )
        if self.add_bias:
            f += (np.zeros_like(t)[:, None],)
        f = np.concatenate(f, axis=1)
        return f


class RffFeatures(BaseFeatures):
    """Quadrature random Fourier features."""

    def __init__(
        self,
        time_sequence: np.ndarray,
        action_dimension: int,
        mean: np.ndarray,
        covariance_in: np.ndarray,
        covariance_out: np.ndarray,
        lengthscale: float,
        order: int,
        sampler,
        use_derivatives,
        add_bias=False,
        limiter=null_limiter,
        *args,
        **kwargs,
    ):
        self.dim_features = time_sequence.shape[0]
        assert lengthscale > 0.0
        assert mean.shape == (action_dimension,)
        assert covariance_in.shape == (1,)
        assert covariance_out.shape == (action_dimension, action_dimension)
        self.add_bias = add_bias
        self.ls = lengthscale
        self.order = order

        x, w = np.polynomial.hermite.hermgauss(2 * order)
        self.x = np.sqrt(2) * x[order:] / self.ls
        self.w = 2 * w[order:] / np.sqrt(np.pi)
        self.dim_features = order * 2
        if self.add_bias:
            self.dim_features += 1
        mean_ = np.zeros((self.dim_features, action_dimension))

        self.mean_fn = mean
        covariance_in_ = covariance_in.item() * np.eye(self.dim_features)
        super().__init__(
            time_sequence,
            action_dimension,
            mean_,
            covariance_in_,
            covariance_out,
            sampler,
            limiter,
            use_derivatives,
        )

    def feat(self, t):
        ph_ = np.einsum("p,n->np", self.x, t)
        f = (
            np.einsum("np,p->np", np.cos(ph_), np.sqrt(self.w)),
            np.einsum("np,p->np", np.sin(ph_), np.sqrt(self.w)),
        )
        if self.add_bias:
            f += (np.ones_like(t)[:, None],)
        f = np.concatenate(f, axis=1)
        return f

    def dfeat(self, t):
        ph_ = np.einsum("p,n->np", self.x, t)
        f = (
            # element-wise frequency scaling
            np.einsum("np,p->np", -np.sin(ph_), self.x * np.sqrt(self.w)),
            np.einsum("np,p->np", np.cos(ph_), self.x * np.sqrt(self.w)),
        )
        if self.add_bias:
            f += (np.zeros_like(t)[:, None],)
        f = np.concatenate(f, axis=1)
        return f


class BaseKernel(BasePrimitive):
    """Kernel-based policies."""

    mean_fn: np.ndarray
    param_bounds: Tuple
    covariance_in_prior: np.ndarray = None
    covariance_in_prior_inv: np.ndarray = None

    @staticmethod
    @abstractmethod
    def k(t_1, t_2):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _k(t_1, t_2, *args):
        raise NotImplementedError

    @abstractproperty
    def param(self):
        raise NotImplementedError

    @param.setter
    def param(self, params):
        pass

    def reset_covariance(self):
        self.covariance_in = self.k(self.t, self.t)
        self.covariance_in_sqrt = np.linalg.cholesky(self.covariance_in)

    def optimize_hyperparams(self, target_matrix):
        """idea is to fit the mean with better hyperparams"""
        target_vec = vec(target_matrix)[:, 0]
        mean = np.zeros((self.dim_features * self.dim_out))

        def objective(param):
            cov = np.kron(self.covariance_out, self._k(self.t, self.t, *param))
            return -multivariate_normal.logpdf(target_vec, mean=mean, cov=cov)

        params_init = np.ones_like(np.array(self.param))
        res = minimize(
            objective, x0=params_init, bounds=self.param_bounds, method="L-BFGS-B"
        )
        logging.info(res)
        param_old = self.param
        self.param = res.x
        logging.info(
            f"Optimized params: {self.param}, was {param_old}, init: {params_init} bounds: {self.param_bounds}"
        )

    def loglikelihood(self, x):
        n, d_i, d_o = x.shape
        assert d_i == self.dim_features
        assert d_o == self.dim_out
        covariance_out_sqrt = np.linalg.cholesky(self.covariance_out)
        covariance_in_inv = np.linalg.solve(
            self.covariance_in_sqrt, self.covariance_in_sqrt.T
        ).T
        covariance_out_inv = np.linalg.solve(
            covariance_out_sqrt, covariance_out_sqrt.T
        ).T
        diff = x - self.mean[None, ...] - self.mean_fn[None, None, :]
        op = np.einsum("bij,ik,bkl->bjl", diff, covariance_in_inv, diff)
        llh = -0.5 * np.trace(np.einsum("ij,bjk->bik", covariance_out_inv, op)) / n
        llh -= 0.5 * self.dim_sample * np.log(2 * np.pi)
        llh -= 0.5 * self.dim_out * np.linalg.slogdet(self.covariance_in)[1]
        llh -= 0.5 * self.dim_features * np.linalg.slogdet(self.covariance_out)[1]
        return llh

    def __call__(self, n_samples) -> (np.ndarray, np.ndarray):
        zs = self.base_sample(n_samples)
        xs = (
            self.mean_fn[None, None, :]
            + self.mean[None, :, :]
            + np.einsum(
                "bij, ki, jl->bkl", zs, self.covariance_in_sqrt, self.cov_out_sqrt.T
            )
        )
        xs_ = self.limiter(xs)
        return xs_, xs_

    def compute_prior(self, plan_time):
        """When plan_time is constant, we compute the prior and inverse before
        execution to save computation.
        """
        self.t = plan_time
        self.covariance_in_prior = self.k(plan_time, plan_time)
        self.covariance_in_prior_inv = np.linalg.inv(self.covariance_in_prior)

    def timesteps_match(self, t):
        if t.shape == self.t.shape:
            if (t == self.t).all():
                return True
        return False

    def update_timesteps(self, time_sequence, anneal=1.0, eps=1e-5):
        assert 0.0 <= anneal <= 1.0
        if not self.timesteps_match(time_sequence):
            d_t = time_sequence.shape[0]
            # precomputed at start, otherwise update if needs be
            if d_t != self.covariance_in_prior.shape[0]:
                self.covariance_in_prior = self.k(self.t, self.t)
                self.covariance_in_prior_inv = np.linalg.inv(self.covariance_in_prior)

            K = (
                self.covariance_in_prior_inv
                @ (self.covariance_in_prior - self.covariance_in)
                @ self.covariance_in_prior_inv.T
            )

            covariance_in_cross = self.k(time_sequence, self.t)
            # this mean is inside the mean function mean_fn, so handle limits carefully
            mean_new = covariance_in_cross @ self.covariance_in_prior_inv @ self.mean
            mean_clipped = self.limiter(mean_new + self.mean_fn[None, :])
            mean_new = mean_clipped - self.mean_fn[None, :]

            covariance_in_new = (
                self.k(time_sequence, time_sequence)
                - anneal * covariance_in_cross @ K @ covariance_in_cross.T
                + eps * self.sigma * np.eye(d_t)
            )
            self.mean = mean_new
            self.covariance_in = covariance_in_new
            self.covariance_in_sqrt = np.linalg.cholesky(covariance_in_new)
            self.t = time_sequence
            self.dim_features = d_t

    def _condition(self, t, action):
        cov_0 = self.covariance_in
        cov_p = self.k(t, t)
        cov_tp = self.k(self.t, t)
        covariance_in = cov_0 - cov_tp @ np.linalg.solve(cov_p, cov_tp.T)
        # n_features X actions
        mean = cov_tp @ np.linalg.solve(cov_p, action - self.mean_fn[None, :])
        self.mean = mean
        self.covariance_in = covariance_in
        self.covariance_in_sqrt = np.linalg.cholesky(covariance_in)

    def weighted_update(self, log_weights, samples, update_covariance_in=True):
        mean, covariance_in, covariance_in_sqrt, covariance_out = (
            self.mean.copy(),
            self.covariance_in.copy(),
            self.covariance_in_sqrt.copy(),
            self.covariance_out.copy(),
        )
        self.map_sequence = samples[np.argmax(log_weights), ...]
        corrected_samples = samples - self.mean_fn[None, None, :]
        mean_, covariance_in_, covariance_out_, ess = m_projection_mavn(
            log_weights, corrected_samples, covariance_in, covariance_out
        )
        self.mean = mean_
        try:
            if update_covariance_in:
                covariance_in_sqrt_ = np.linalg.cholesky(covariance_in_)
                self.covariance_in = covariance_in_
                self.covariance_in_sqrt = covariance_in_sqrt_
            kl = 0.0  # comment out to save computation for experiments
            # kl = matrix_gaussian_kl(
            #     self.mean,
            #     self.covariance_in,
            #     self.covariance_out,
            #     mean,
            #     covariance_in,
            #     self.covariance_out,
            # )

        except np.linalg.LinAlgError:
            logging.warning(f"Update error, ESS={ess}")
            logging.warning(f"weights: [{log_weights.max()}, {log_weights.min()}]")
            eigvals = np.linalg.eigvalsh(covariance_in_)
            logging.warning(f"Eigvals: [{eigvals.max()}, {eigvals.min()}]")
            logging.exception("Weighted update error, reverting")
            kl = 0.0
            covariance_in += SIGMA_MIN * np.eye(self.dim_features)
            covariance_in_sqrt += SIGMA_MIN * np.eye(self.dim_features)
            ess = samples.shape[0]
            (self.covariance_in, self.covariance_in_sqrt, self.covariance_out,) = (
                covariance_in,
                covariance_in_sqrt,
                covariance_out,
            )

        return ess, kl

    def predict(self, only_mean=False):
        mu_y = self.mean_fn[None, :] + self.mean
        if only_mean:
            return mu_y
        else:
            sigma_y_in = self.covariance_in
            sigma_y_out = self.covariance_out
            sigma_y = np.sqrt(
                np.diag(np.kron(sigma_y_out, sigma_y_in)).reshape(mu_y.shape)
            )
            return mu_y, sigma_y_in, sigma_y_out, sigma_y


class StationaryKernel(BaseKernel):
    """Main class of kernels we want: squared exponential, Matern, white noise,etc."""

    param_bounds = ((1e-5, None), (1e-3, 1e3))

    def __init__(
        self,
        time_sequence: np.ndarray,
        action_dimension: int,
        mean: np.ndarray,
        covariance_in: np.ndarray,
        covariance_out: np.ndarray,
        lengthscale: float,
        sampler,
        limiter=null_limiter,
        *args,
        **kwargs,
    ):
        self.dim_features = time_sequence.shape[0]
        assert lengthscale > 0.0
        assert mean.shape == (action_dimension,), f"{mean.shape} != (action_dimension,)"
        assert covariance_in.shape == (1,)
        assert covariance_out.shape == (action_dimension, action_dimension)
        self.mean_fn = mean
        self.ls = lengthscale
        self.sigma = covariance_in.item()
        mean_ = np.zeros((self.dim_features, action_dimension))
        # assert covariance_in.shape == (self.dim_features, self.dim_features)
        covariance_in = self.k(time_sequence, time_sequence)
        super().__init__(
            time_sequence,
            action_dimension,
            mean_,
            covariance_in,
            covariance_out,
            sampler,
            limiter,
        )

    @property
    def param(self):
        return self.sigma, self.ls

    @param.setter
    def param(self, params):
        assert params > 0.0
        self.sigma, self.ls = params

    def k(self, t1, t2):
        return self._k(t1, t2, sigma=self.sigma, ls=self.ls)

    @staticmethod
    def _k(t1, t2, sigma, ls, lib):
        raise NotImplementedError


class SquaredExponentialKernel(StationaryKernel):
    can_optimize = True
    param_bounds = ((1e-5, None), (1e-5, 1e3))

    @staticmethod
    def _k(t1, t2, sigma, ls, lib=np):
        t1 = t1[:, None] if len(t1.shape) == 1 else t1
        t2 = t2[:, None] if len(t2.shape) == 1 else t2
        err = t1 - t2.T
        k = sigma * lib.exp(-0.5 * lib.power(err / ls, 2))
        d1, d2 = k.shape
        if d1 == d2:
            k += 1e-3 * sigma * np.eye(d1)
        return k


class Matern12Kernel(StationaryKernel):
    @staticmethod
    def _k(t1, t2, sigma, ls, eps=1e-8, lib=np):
        t1 = t1[:, None] if len(t1.shape) == 1 else t1
        t2 = t2[:, None] if len(t2.shape) == 1 else t2
        abs_diff = lib.abs(t1 - t2.T)
        abs_diff[abs_diff == 0.0] = eps
        k = sigma * lib.exp(-abs_diff / ls)
        return k


SQRT3 = np.sqrt(3)


class Matern32Kernel(StationaryKernel):
    @staticmethod
    def _k(t1, t2, sigma, ls, eps=1e-8, lib=np):

        t1 = t1[:, None] if len(t1.shape) == 1 else t1
        t2 = t2[:, None] if len(t2.shape) == 1 else t2
        abs_diff = lib.abs(t1 - t2.T)
        abs_diff[abs_diff == 0.0] = eps
        d = SQRT3 * abs_diff / ls
        k = sigma * (1.0 + d) * lib.exp(-d)
        return k


SQRT5 = np.sqrt(5)


class Matern52Kernel(StationaryKernel):
    @staticmethod
    def _k(t1, t2, sigma, ls, eps=1e-8, lib=np):
        t1 = t1[:, None] if len(t1.shape) == 1 else t1
        t2 = t2[:, None] if len(t2.shape) == 1 else t2
        abs_diff = lib.abs(t1 - t2.T)
        abs_diff[abs_diff == 0.0] = eps
        d = SQRT5 * abs_diff / ls
        k = sigma * (1.0 + d + d ** 2 / 3) * lib.exp(-d)
        return k


class PeriodicKernel(StationaryKernel):

    # self.sigma, self.ls, self.period,
    param_bounds = ((1e-3, None), (1e-4, 1e3), (1e-3, 1e3))

    def __init__(
        self,
        time_sequence: np.ndarray,
        action_dimension: int,
        mean: np.ndarray,
        covariance_in: np.ndarray,
        covariance_out: np.ndarray,
        lengthscale: float,
        period: float,
        sampler,
        limiter=null_limiter,
        use_derivatives=False,
        *args,
        **kwargs,
    ):
        self.period = period
        super().__init__(
            time_sequence,
            action_dimension,
            mean,
            covariance_in,
            covariance_out,
            lengthscale,
            sampler,
            limiter,
            # use_derivatives,
            args,
            kwargs,
        )

    @property
    def param(self):
        return self.sigma, self.ls, self.period

    @param.setter
    def param(self, params):
        (sigma, ls, p,) = params
        assert ls > 0.0
        assert p > 0.0
        self.sigma, self.ls, self.period = sigma, ls, p

    def k(self, t1, t2):
        return self._k(t1, t2, sigma=self.sigma, ls=self.ls, per=self.period)

    @staticmethod
    def _k(t1, t2, sigma, ls, per, eps=1e-8, lib=np):
        t1 = t1[:, None] if len(t1.shape) == 1 else t1
        t2 = t2[:, None] if len(t2.shape) == 1 else t2
        diff = t1 - t2.T
        diff[diff == 0.0] = -eps
        abs_diff = lib.abs(diff)
        k_per = sigma * lib.exp(-2 * lib.sin(np.pi * abs_diff / per) ** 2 / ls)
        d1, d2 = k_per.shape
        if d1 == d2:
            k_per += 1e-3 * sigma * np.eye(d1)
        return k_per


class WhiteNoiseKernel(BaseKernel):

    param_bounds = ((1e-5, None),)

    def __init__(
        self,
        time_sequence: np.ndarray,
        action_dimension: int,
        mean: np.ndarray,
        covariance_in: np.ndarray,
        covariance_out: np.ndarray,
        sampler,
        use_derivatives=False,
        limiter=null_limiter,
        *args,
        **kwargs,
    ):
        self.dim_features = time_sequence.shape[0]
        assert mean.shape == (action_dimension,), f"{mean.shape} != (action_dimension,)"
        assert covariance_in.shape == (1,)
        assert covariance_out.shape == (action_dimension, action_dimension)
        self.mean_fn = mean
        self.sigma = covariance_in.item()
        mean_ = np.zeros((self.dim_features, action_dimension))
        covariance_in = self.k(time_sequence, time_sequence)
        super().__init__(
            time_sequence,
            action_dimension,
            mean_,
            covariance_in,
            covariance_out,
            sampler,
            limiter,
            use_derivatives,
        )

    @property
    def param(self):
        return self.sigma

    @param.setter
    def param(self, params):
        self.sigma = params[0]

    def k(self, t1, t2):
        return self._k(t1, t2, self.sigma)

    @staticmethod
    def _k(t1, t2, sigma, lib=np):
        t1, t2 = t1.reshape((-1, 1)), t2.reshape((-1, 1))
        err = t1 - t2.T
        return sigma * (err == 0.0)

    def update_timesteps(self, time_sequence, eps=1e-7):
        if not self.timesteps_match(time_sequence):
            t1, t2 = self.t.reshape((-1, 1)), time_sequence.reshape((-1, 1))
            d_t = time_sequence.shape[0]
            remap = 1.0 * ((t2 - t1.T) == 0.0)
            cov_new = self.k(time_sequence, time_sequence)
            self.mean = remap @ self.mean
            self.covariance_in = remap @ self.covariance_in @ remap.T
            self.covariance_in += (np.eye(d_t) - remap @ remap.T) @ cov_new
            self.covariance_in_sqrt = np.linalg.cholesky(self.covariance_in)
            self.t = time_sequence
            self.dim_features = d_t


class WhiteNoiseIid(object):

    can_optimize = False
    map_sequence: np.ndarray

    def __init__(
        self,
        time_sequence: np.ndarray,
        action_dimension: int,
        mean: np.ndarray,
        covariance_in: np.ndarray,
        covariance_out: np.ndarray,
        use_derivatives=False,
        limiter=null_limiter,
        *args,
        **kwargs,
    ):
        self.dim_features = time_sequence.shape[0]
        self.dim_out = action_dimension
        assert mean.shape == (action_dimension,), f"{mean.shape} != (action_dimension,)"
        assert covariance_in.shape == (1,)
        assert covariance_out.shape == (action_dimension, action_dimension)
        self.t = time_sequence
        self.sigma = np.sqrt(covariance_out * covariance_in.item())
        self.mean_fn = mean
        self.mean = np.zeros((self.dim_features, action_dimension))
        self.std = np.ones((self.dim_features, action_dimension)) @ self.sigma
        self.limiter = limiter

    def compute_prior(self, t):
        pass

    def reset_covariance(self):
        self.std = np.ones((self.dim_features, self.dim_out)) @ self.sigma

    @property
    def covariance_out(self):
        return np.diag(self.std.mean(axis=0) ** 2)

    @property
    def entropy(self):
        return multivariate_gaussian_entropy(
            np.diag(self.std.flatten() ** 2), self.dim_out * self.dim_features
        )

    def weighted_update(self, log_weights, samples, update_covariance_in=True):
        self.map_sequence = samples[np.argmax(log_weights), ...]
        log_nw = log_weights - logsumexp(log_weights)
        nw = np.exp(log_nw)
        ess = np.exp(-logsumexp(2 * log_nw))
        corrected_samples = samples - self.mean_fn[None, None, :]
        mean = np.einsum("b,bij->ij", nw, corrected_samples)
        diff = corrected_samples - mean[None, ...]
        self.mean = mean
        if update_covariance_in:
            self.std = np.sqrt(np.einsum("b,bij->ij", nw, diff ** 2))
        kl = 0.0
        return ess, kl

    def __call__(self, n_samples: int) -> (np.ndarray, np.ndarray):
        zs = np.random.randn(n_samples, self.dim_features, self.dim_out)
        xs = (
            self.mean_fn[None, None, :]
            + self.mean[None, ...]
            + np.einsum("ij,bij->bij", self.std, zs)
        )
        xs_ = self.limiter(xs)
        return xs_, xs_

    def update_timesteps(self, time_sequence, anneal=1.0, eps=1e-7):
        t1, t2 = self.t.reshape((-1, 1)), time_sequence.reshape((-1, 1))
        d_t = time_sequence.shape[0]
        remap = 1.0 * ((t2 - t1.T) == 0.0)
        std_new = np.ones((d_t, self.dim_out)) @ self.sigma
        self.mean = remap @ self.mean
        std = np.sqrt(
            np.power(remap @ self.std, 2)
            + np.power((np.eye(d_t) - remap @ remap.T) @ std_new, 2)
        )
        self.std = anneal * std + (1 - anneal) * std_new
        self.t = time_sequence
        self.dim_features = d_t

    def predict(self, only_mean=False):
        mean = self.mean_fn[None, :] + self.mean
        if only_mean:
            return mean
        else:
            return mean, self.std ** 2


class ColouredNoise(WhiteNoiseIid):

    def __init__(
        self,
        time_sequence: np.ndarray,
        action_dimension: int,
        mean: np.ndarray,
        covariance_in: np.ndarray,
        covariance_out: np.ndarray,
        sampler,
        use_derivatives=False,
        beta=2.0,
        limiter=null_limiter,
        *args,
        **kwargs,
    ):
        self.beta = beta
        self.sampler = sampler(0)
        super().__init__(
            time_sequence,
            action_dimension,
            mean,
            covariance_in,
            covariance_out,
            use_derivatives,
            limiter,
            args,
            kwargs,
        )

    def update_timesteps(self, time_sequence, anneal=1.0, eps=1e-7):
        super().update_timesteps(time_sequence, anneal)
        if isinstance(self.sampler, samplers.Particles):
            if self.sampler.has_particles:
                particles = np.copy(self.sampler.particles)
                last_action = particles[:, -1, :]
                particles[:, :-1, :] = self.sampler.particles[:, 1:, :]
                particles[:, -1, :] = last_action
                self.sampler.particles = particles

    def __call__(self, n_samples) -> (np.ndarray, np.ndarray):
        #  temporal correlations are in last axis
        if self.dim_features > 1:
            zs = colorednoise.powerlaw_psd_gaussian(
                self.beta, size=(n_samples, self.dim_out, self.dim_features)
            ).transpose([0, 2, 1])
        else:
            zs = np.random.randn(n_samples, self.dim_features, self.dim_out)
        if hasattr(self.sampler, "particles"):
            zs = self.sampler.add_particles(zs)
        xs = (
            self.mean_fn[None, None, :]
            + self.mean[None, ...]
            + np.einsum("ij,bij->bij", self.std, zs)
        )
        xs_ = self.limiter(xs)
        return xs_, xs_


def convolve(signal_samples: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    n, h, d = signal_samples.shape
    w = kernel.shape[0]
    w_ = w - 1
    assert w > 1
    smoothed = signal_samples.copy()
    if h > w_:
        for t in range(w_, h):
            smoothed[:, t, :] = np.einsum(
                "bij,i->bj", smoothed[:, t - w_ : t + 1, :], kernel
            )
    return smoothed


class SmoothExplorationNoise(WhiteNoiseIid):

    def __init__(
        self,
        time_sequence: np.ndarray,
        action_dimension: int,
        mean: np.ndarray,
        covariance_in: np.ndarray,
        covariance_out: np.ndarray,
        sampler,
        beta=2.0,
        limiter=null_limiter,
        use_derivatives=False,
        *args,
        **kwargs,
    ):
        assert 0.0 < beta < 1.0, f"beta is {beta}"
        self.beta = beta
        self.filter_coef = np.array([1 - beta, beta])
        super().__init__(
            time_sequence,
            action_dimension,
            mean,
            covariance_in,
            covariance_out,
            use_derivatives,
            limiter,
            args,
            kwargs,
        )

    def __call__(self, n_samples: int) -> (np.ndarray, np.ndarray):
        zs = np.random.randn(n_samples, self.dim_features, self.dim_out)
        zs_smooth = convolve(zs, self.filter_coef)
        xs = (
            self.mean_fn[None, None, :]
            + self.mean[None, ...]
            + np.einsum("ij,bij->bij", self.std, zs_smooth)
        )
        xs_ = self.limiter(xs)
        return xs_, xs_


class SmoothActionNoise(SmoothExplorationNoise):
    """The often implemented version of SmoothExplorationNoise that also smooths the mean."""

    def __call__(self, n_samples: int) -> (np.ndarray, np.ndarray):
        zs = np.random.randn(n_samples, self.dim_features, self.dim_out)
        xs = (
            self.mean_fn[None, None, :]
            + self.mean[None, ...]
            + np.einsum("ij,bij->bij", self.std, zs)
        )
        xs_smooth = convolve(xs, self.filter_coef)
        xs_smooth = self.limiter(xs_smooth)
        return xs_smooth, xs_smooth


class LinearGaussianDynamicalSystemKernel(BaseKernel):
    """."""

    can_optimize = False

    def __init__(
        self,
        time_sequence: np.ndarray,
        action_dimension: int,
        mean: np.ndarray,
        covariance_in: np.ndarray,
        covariance_out: np.ndarray,
        order: int,
        sampler,
        use_derivatives,
        limiter=null_limiter,
        *args,
        **kwargs,
    ):
        self.dim_features = time_sequence.shape[0]
        assert order in [1, 2, 3]
        assert mean.shape == (action_dimension,)
        assert covariance_in.shape == (1,)
        self.mean_fn = mean
        self.order = order
        self.sigma = covariance_in.item()
        self.Q = np.zeros((order, order))
        self.Q[-1, -1] = self.sigma
        mean_ = np.zeros((self.dim_features, action_dimension))
        # assert covariance_in.shape == (self.dim_features, self.dim_features)
        covariance_in = self.k(time_sequence, time_sequence)
        super().__init__(
            time_sequence,
            action_dimension,
            mean_,
            covariance_in,
            covariance_out,
            sampler,
            limiter,
            use_derivatives,
        )

    def k(self, t1, t2):
        # TODO implement properly? Needs a bit of love
        # see https://github.com/gtrll/gpmp2/blob/main/gpmp2/gp/GaussianProcessPriorLinear.h
        N = t1.shape[0]
        # assume constant timestep
        A = self.transition_matrx(t1[1], t1[0], d=self.order)
        A_ = np.kron(np.eye(N, k=0), np.eye(self.order))
        # where did this come from?
        for i in range(1, N):
            A_ += np.kron(np.eye(N, k=-i), np.linalg.matrix_power(A, i))
        Q_ = block_diag(*([1e-3 * np.eye(self.order),] + [self.Q,] * (N - 1)))
        disturbance = block_diag(*([1e-6 * np.eye(self.order),] * N))
        K = A_ @ Q_ @ A_.T + disturbance
        return K[:: self.order, :: self.order]

    def _condition(self, t, action):
        idx = (t == self.t).nonzero()[0]
        cov_0 = self.covariance_in
        cov_p = cov_0[np.ix_(idx, idx)]
        cov_tp = cov_0[:, idx]
        mean = cov_tp @ np.linalg.solve(cov_p, action - self.mean_fn[None, :])
        covariance_in = cov_0 - cov_tp @ np.linalg.solve(cov_p, cov_tp.T)
        self.mean = mean
        self.covariance_in = covariance_in

    @staticmethod
    def transition_matrx(t2, t1, d=3):
        A = np.eye(d)
        dt = t2 - t1
        if d == 3:
            A[0, 1] = dt
            A[0, 2] = 0.5 * dt ** 2
            A[1, 2] = dt
        elif d == 2:
            A[0, 1] = dt
        elif d == 1:
            pass
        else:
            raise ValueError("Only defined for d = 1-3")
        return A
