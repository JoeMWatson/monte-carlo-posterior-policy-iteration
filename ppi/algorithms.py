"""
Posterior policy iteration methods and related algorithms.
"""

import traceback
from typing import Callable, Tuple, List
from warnings import warn

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm as _tqdm

import ppi.samplers as samplers
from ppi.policies import multivariate_gaussian_entropy, multivariate_gaussian_kl

__all__ = [
    "Ais",
    "Cem",
    "iCem",
    "Reps",
    "Lbps",
    "More",
    "Essps",
    "Mppi",
    "MppiUpdateCovariance",
]


EPS = np.finfo(np.float64).tiny

COVAR_REG = 1e-20

alpha_lower = 1e-5
alpha_upper = 5e2


def null_callback(
    iteration: int, f: Callable, actions: np.ndarray, costs: np.ndarray, policy
):
    """Used to no-op during step function"""
    return False


class Base(object):
    """Base class for optimizers."""

    trace = {}  # for optimization telemetry

    @classmethod
    def reset(cls, policy):
        """Reset telemetry."""
        cls.trace = {"mean": [], "std": [], "ent": []}

    @staticmethod
    def filter(costs: List[float], weights: List[np.ndarray]):
        """Filter out NaN returns."""
        bad_idx = np.isnan(costs)
        good_idx = ~bad_idx
        if bad_idx.all():
            warn("All trajectories entered absorbing state. Not filtering.")
            costs_, weights_ = np.zeros_like(costs), weights
        else:
            costs_ = costs[good_idx]
            weights_ = weights[good_idx, ...]
        return costs_, weights_

    def __call__(
        self,
        f: Callable,
        policy,
        n_samples: int,
        n_iters: int,
        callback=null_callback,
        use_tqdm=True,
    ):
        """Evaluate function f with policy samples."""
        self.reset(policy)
        iterations = range(n_iters)
        iterations = _tqdm(iterations) if use_tqdm else iterations
        self.trace["ent"] += [policy.entropy]
        for i in iterations:
            actions, weights = policy(n_samples)
            costs = f(actions)
            costs, weights = self.filter(costs, weights)
            policy = self.update(costs, weights, policy)
            self.trace["mean"].append(costs.mean())
            self.trace["std"].append(costs.std())
            flag = callback(i, f, actions, costs, policy)
            if flag:
                break
        return self.trace

    def update(self, costs: np.ndarray, samples: np.ndarray, policy):
        """Update policy based on cost-weighted samples."""
        raise NotImplemented


class Cem(Base):
    """Cross-entropy method."""

    name = "CEM"

    def __init__(self, n_elites: int, *args, **kwargs):
        self.n_elites = n_elites

    def reset(self, policy):
        super().reset(policy)
        self.trace["ess"] = []
        self.trace["kl"] = []
        self.trace["ent"] = []
        self.trace["weight_ent"] = []
        policy.reset_covariance()

    def update(self, costs: np.ndarray, samples: np.ndarray, policy):
        idxs = np.argsort(costs)[: self.n_elites]
        log_w = -1e12 * np.ones_like(costs)
        log_w[idxs] = 0.0
        ess, kl = policy.weighted_update(log_w, samples)
        policy.map_sequence = samples[idxs[0], ...]
        log_nw = log_w - logsumexp(log_w)
        self.trace["ess"] += [ess]
        self.trace["kl"] += [kl]
        self.trace["ent"] += [policy.entropy]
        self.trace["weight_ent"] += [(log_nw * np.exp(log_nw)).sum()]
        return policy


class iCem(Base):
    """'Improved' cross-entropy method."""

    name = "iCEM"

    def __init__(self, n_elites: int, sample_reuse_pc=0.33, *args, **kwargs):
        self.n_elites = n_elites
        self.sample_reuse_pc = sample_reuse_pc
        self.n_reuse = int(sample_reuse_pc * n_elites)

    def reset(self, policy):
        super().reset(policy)
        self.trace["ess"] = []
        self.trace["kl"] = []
        self.trace["ent"] = []
        self.trace["weight_ent"] = []
        policy.reset_covariance()

    def update(self, costs: np.ndarray, samples: np.ndarray, policy):
        srt_idx = np.argsort(costs)
        elite_idxs = srt_idx[: self.n_elites]
        reuse_idxs = srt_idx[: self.n_reuse]
        log_w = -1e12 * np.ones_like(costs)
        log_w[elite_idxs] = 0.0
        ess, kl = policy.weighted_update(log_w, samples)
        policy.map_sequence = samples[elite_idxs[0], ...]
        self.trace["ess"] += [ess]
        self.trace["kl"] += [kl]
        self.trace["ent"] += [policy.entropy]
        log_nw = log_w - logsumexp(log_w)
        self.trace["weight_ent"] += [(log_nw * np.exp(log_nw)).sum()]
        if isinstance(policy.sampler, samplers.Particles):
            policy.sampler.particles = samples[reuse_idxs, ...]
            assert policy.sampler.has_particles
        return policy


class Reps(Base):
    """Relative entropy policy search."""

    name = "REPS"

    def __init__(self, epsilon: float, *args, **kwargs):
        self.epsilon = epsilon

    def reset(self, policy):
        super().reset(policy)
        self.trace["ess"] = []
        self.trace["kl"] = []
        self.trace["alpha"] = []
        self.trace["ent"] = []

    def update(self, costs: np.ndarray, samples: np.ndarray, policy):
        costs_ = (costs - np.min(costs)) / (np.max(costs) - np.min(costs) + EPS)

        def dual(alpha_in):
            try:
                alpha = alpha_in.item()
                w_ = np.exp(-alpha * costs_)
                g = self.epsilon / alpha + np.log(w_.mean()) / alpha
                return g
            except Exception as exc:
                print(alpha_in)
                raise exc

        def dual_grad(alpha_in):
            alpha = alpha_in.item()
            w_ = np.exp(-alpha * costs_)
            a2 = alpha ** -2
            ndg = a2 * (self.epsilon + np.log(w_.mean())) + np.einsum(
                "b,b->", w_, costs_
            ) / (alpha * w_.sum())
            return -ndg

        res = minimize(
            dual,
            jac=dual_grad,
            method="L-BFGS-B",
            bounds=((alpha_lower, alpha_upper),),
            x0=np.array([1.0]),
            options={"maxiter": 1000},
        )

        alpha = res.x
        log_w_ = -alpha * costs_
        ess, kl = policy.weighted_update(log_w_, samples)
        self.trace["ess"].append(ess)
        self.trace["alpha"].append(alpha)
        self.trace["kl"].append(kl)
        self.trace["ent"].append(policy.entropy)
        return policy


class More(Base):
    """Model-based stochastic search."""

    name = "MORE"

    def __init__(
        self,
        epsilon: float,
        base_entropy: float,
        entropy_rate: float,
        dimension: int,
        ridge_coeff=1e-5,
        *args,
        **kwargs,
    ):
        assert epsilon > 0
        assert base_entropy
        assert entropy_rate >= 0
        self.epsilon = epsilon
        self.base_entropy = base_entropy
        self.entropy_rate = entropy_rate
        self.features = PolynomialFeatures(2)
        self.model_fitter = Ridge(alpha=ridge_coeff, fit_intercept=False)
        self.dimension = dimension

    def reset(self, policy):
        super().reset(policy)
        self.trace["alpha"] = []
        self.trace["omega"] = []
        self.trace["kl"] = []
        self.trace["ent"] = []
        self.trace["ess"] = []
        self.trace["fit_lin"] = []
        self.trace["fit"] = []

    @staticmethod
    def F(Q, R_inv, eta):
        # in the paper F = (eta * Q^{-1} - 2R)^{-1}
        # is equal to alpha * Q - alpha^2 * Q (alpha Q - 0.5 * R^{-1})^{-1} Q
        # as alpha = 1 / eta
        alpha = 1 / eta
        try:
            return alpha * Q - alpha ** 2 * Q @ np.linalg.solve(
                alpha * Q - 0.5 * R_inv, Q
            )
        except np.linalg.LinAlgError:
            # use low rank approximation of inverse
            s, v = np.linalg.eigh(-0.5 * R_inv)
            pos_idx = np.argwhere(s > 0.0).flatten()
            n_pos = pos_idx.shape[0]
            print(f"F is singular! {n_pos} good eigenvalues")
            v_ = v[:, pos_idx]
            aQ_inv = np.linalg.inv(alpha * Q)
            A_inv = aQ_inv - aQ_inv @ v_ @ np.linalg.solve(
                np.diag(1 / s[pos_idx]) + v_.T @ aQ_inv @ v_, v_.T @ aQ_inv
            )
            return alpha * Q - alpha ** 2 * Q @ A_inv @ Q

    @staticmethod
    def f(Q: np.ndarray, b: np.ndarray, r: np.ndarray, eta: float):
        return np.linalg.solve(Q, b) * eta + r

    def fit_quadratic_model(self, w: np.ndarray, c: np.ndarray):
        dim_w = self.dimension
        feat = self.features.fit_transform(w)
        self.model_fitter.fit(feat, c)
        y_ = self.model_fitter.predict(feat)
        rmse_fit = np.sqrt(np.power(c - y_, 2).mean())
        param = self.model_fitter.coef_[:]

        # quadratic / cross-terms
        uid = np.triu_indices(dim_w)
        R = np.zeros((dim_w, dim_w))
        R[uid] = param[1 + dim_w :]
        R.T[uid] = R[uid]
        R_diag = np.diag(np.diag(R))
        # halve off-diagonal features
        R = 0.5 * (R - R_diag) + R_diag

        # linear terms (mean)
        r = param[1 : 1 + dim_w]
        # offset
        r0 = param[0]
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError as exc:
            reg_ = 1e-9
            s, v = np.linalg.eigh(R)
            neg_idx = np.argwhere(s < 0.0).flatten()
            n_neg = neg_idx.shape[0]
            print(
                f"Quadratic weights are singular! {self.dimension - n_neg} bad eigenvalues"
            )
            v_ = v[:, neg_idx]
            R = v_ @ np.diag(s[neg_idx]) @ v_.T
            R = 0.5 * (R + R.T)
            # compute inverse of rank deficient matrix via Sherman-Morrison-Woodbury
            reg_inv = -np.eye(self.dimension) / reg_
            R_inv = reg_inv - reg_inv @ v_ @ np.linalg.solve(
                np.diag(1 / s[neg_idx]) + v_.T @ reg_inv @ v_, v_.T @ reg_inv
            )

        y_ = np.einsum("bi,bj,ij->b", w, w, R) + w @ r + r0
        rmse = np.sqrt(np.power(c - y_, 2).mean())
        self.trace["fit"].append(rmse)
        self.trace["fit_lin"].append(rmse_fit)

        return r0, r, R, R_inv

    def update(self, costs: np.ndarray, samples: np.ndarray, policy):
        assert (
            len(samples.shape) == 2
        ), "More is defined for vector parameters, not matrices"
        rewards = -1.0 * costs
        rewards -= rewards.max()
        rewards /= np.abs(rewards).max()
        rewards *= 100.0

        r0, r, R, R_inv = self.fit_quadratic_model(samples, rewards)

        # to match equations
        b, Q = policy.mu, policy.sigma
        ent_n = multivariate_gaussian_entropy(Q, self.dimension)
        beta = self.entropy_rate * (ent_n - self.base_entropy) + self.base_entropy

        def dual(eta_omega_in: Tuple):
            """
            Eq. 4 of Neurips paper
            """
            try:
                eta, omega = eta_omega_in
                F = More.F(Q, R_inv, eta)
                f = More.f(Q, b, r, eta)  # large
                fFf = f.T @ F @ f
                bQb = b.T @ np.linalg.solve(Q, b)
                eta_omega = omega + eta
                ent_Q = np.linalg.slogdet(2 * np.pi * Q)[1]
                ent_F = np.linalg.slogdet(2 * np.pi * eta_omega * F)[1]
                g = (
                    self.epsilon * eta
                    - beta * omega
                    + 0.5 * (fFf - bQb * eta - eta * ent_Q + ent_F * eta_omega)
                )
                return g
            except Exception as exc:
                print(f"Eta Omega: {eta_omega_in}")
                traceback.print_exc()
                raise exc

        def dual_grad(eta_omega_in):
            eta, omega = eta_omega_in

            F = More.F(Q, R_inv, eta)
            f = More.f(Q, b, r, eta)

            eta_omega = omega + eta
            ent_Q = np.linalg.slogdet(2 * np.pi * Q)[1]
            ent_F = np.linalg.slogdet(2 * np.pi * eta_omega * F)[1]

            dF_deta = -F.T @ np.linalg.solve(Q, F)
            df_deta = np.linalg.solve(Q, b)

            eta_grad = self.epsilon + 0.5 * (
                2.0 * f.T @ F @ df_deta
                + f.T @ dF_deta @ f
                - b.T @ df_deta
                - ent_Q
                + ent_F
                + self.dimension
                - eta_omega * np.trace(np.linalg.solve(Q, F).T)
            )
            omega_grad = -beta + 0.5 * (ent_F + self.dimension)
            return np.array([eta_grad.item(), omega_grad])

        res = minimize(
            dual,
            jac=dual_grad,
            method="L-BFGS-B",
            bounds=[(alpha_lower, alpha_upper), (alpha_lower, alpha_upper)],
            x0=np.ones((2,)),
        )
        eta, omega = res.x

        F, f = self.F(Q, R_inv, eta), self.f(Q, b, r, eta)
        mu_f = F @ f
        sigma_f = (eta + omega) * F

        # linear search over interpolation with positive definite guarantee
        t = 1
        success = False
        sigma_f_inv = np.linalg.inv(sigma_f)
        sigma_inv = np.linalg.inv(Q)
        G = sigma_inv - sigma_f_inv
        M = G @ Q @ G
        nu = sigma_inv @ b
        nu_f = sigma_f_inv @ mu_f
        for i in range(3):
            try:
                nu_ = (1 - t) * nu + t * nu_f
                lambda_ = (1 - t) * sigma_inv + t * sigma_f_inv + 0.5 * t ** 2 * M
                sigma_ = np.linalg.inv(lambda_)
                mu_ = sigma_ @ nu_
                kl = multivariate_gaussian_kl(mu_, sigma_, b, Q)
                if kl <= self.epsilon:
                    success = True
                    break
            except np.linalg.LinAlgError as exc:
                print(f"Pos. def. iter {i} singular")
            finally:
                t = 0.5 * t

        if not success:
            policy.smooth_update(mu_, sigma_, 1.0)
        else:
            print("Update not positive definite. Skipped.")

        kl = multivariate_gaussian_kl(mu_, sigma_, b, Q)
        ent = multivariate_gaussian_entropy(sigma_, self.dimension)
        log_w = rewards / eta
        log_nw_ = log_w - logsumexp(log_w)
        ess = np.exp(-logsumexp(2 * log_nw_))

        self.trace["alpha"].append(1 / eta)
        self.trace["omega"].append(omega)
        self.trace["kl"].append(kl)
        self.trace["ent"].append(ent)
        self.trace["ess"].append(ess)

        return policy


class MppiBase(Base):
    """Model predictive path integral control."""

    name = "MPPI"
    update_covariance: bool

    def __init__(self, alpha: float, *args, **kwargs):
        """alpha is the inverse temperature of the likelihood."""
        self.alpha = alpha

    def reset(self, policy):
        super().reset(policy)
        self.trace["ess"] = []
        self.trace["kl"] = []
        self.trace["ent"] = []

    def update(self, costs: np.ndarray, samples: np.ndarray, policy):
        costs_ = costs - np.min(costs)
        log_w = -costs_ * self.alpha
        ess, kl = policy.weighted_update(
            log_w, samples, update_covariance_in=self.update_covariance
        )
        self.trace["ess"].append(ess)
        self.trace["kl"].append(kl)
        self.trace["ent"].append(policy.entropy)
        return policy


class Mppi(MppiBase):
    """MPPI with fixed action covariance."""

    update_covariance = False


class MppiUpdateCovariance(MppiBase):
    """MPPI with updated action covariance."""

    update_covariance = True


class Ais(Base):
    """Adaptive importance sampling."""

    name = "AIS"

    def __init__(self, alpha: float, *args, **kwargs):
        self.alpha = alpha

    def reset(self, policy):
        super().reset(policy)
        self.trace["ess"] = []
        self.trace["kl"] = []
        self.trace["ent"] = []

    def log_w(self, costs: np.ndarray):
        costs_ = (costs - np.min(costs)) / (np.max(costs) - np.min(costs) + EPS)
        return -costs_ * self.alpha, self.alpha

    def update(self, costs: np.ndarray, samples: np.ndarray, policy):
        log_w, _ = self.log_w(costs)
        ess, kl = policy.weighted_update(log_w, samples)
        self.trace["ess"].append(ess)
        self.trace["kl"].append(kl)
        self.trace["ent"].append(policy.entropy)
        return policy


class Lbps(Base):
    """Lower-bound policy search."""

    name = "SNISLB"

    def __init__(self, delta: float, *args, **kwargs):
        self.delta = delta
        self.max = np.NINF

    def reset(self, policy):
        super().reset(policy)
        self.trace["ess"] = []
        self.trace["alpha"] = []
        self.trace["kl"] = []
        self.trace["ent"] = []
        self.trace["max"] = []

    def log_w(self, costs: np.ndarray):
        min_, max_ = np.min(costs), np.max(costs)
        costs_ = (costs - min_) / (max_ - min_ + EPS)

        _max = 1.0  # due to normalization
        lambda_ = _max * np.sqrt((1 - self.delta) / self.delta)

        # lower-bound optimization
        def lower_bound(alpha, eps=1e-6):
            try:
                log_w_ = -alpha * costs_
                log_nw_ = log_w_ - logsumexp(log_w_)
                nw_ = np.exp(log_nw_)
                ess = np.exp(-logsumexp(2 * log_nw_))
                ec = np.einsum("b,b->", nw_, costs_)
                err = lambda_ / np.sqrt(ess)
                return ec + err
            except Exception:
                raise

        res = minimize_scalar(
            lower_bound,
            method="brent",
            bounds=(alpha_lower, alpha_upper),
            options={"maxiter": 5000},
        )
        alpha = res.x
        log_w = -alpha * costs_
        return log_w, alpha

    def update(self, costs: np.ndarray, samples: np.ndarray, policy):
        log_w, alpha = self.log_w(costs)
        ess, kl = policy.weighted_update(log_w, samples)
        self.trace["ess"].append(ess)
        self.trace["kl"].append(kl)
        self.trace["ent"].append(policy.entropy)
        self.trace["alpha"].append(alpha)

        return policy


class Essps(Base):
    """Effective sample size policy search."""

    name = "ESSPS"

    def __init__(self, n_elites: int, *args, **kwargs):
        self.ess = int(n_elites)

    def reset(self, policy):
        super().reset(policy)
        self.trace["ess"] = []
        self.trace["kl"] = []
        self.trace["ent"] = []
        self.trace["weight_ent"] = []
        self.trace["alpha"] = []

    def update(self, costs, samples, policy):
        costs_ = (costs - np.min(costs)) / (np.max(costs) - np.min(costs) + EPS)

        def ess_err(alpha, eps=1e-6):
            try:
                log_w_ = -alpha * costs_
                log_nw_ = log_w_ - logsumexp(log_w_)
                ess = np.exp(-logsumexp(2 * log_nw_))
                return np.abs(ess - self.ess)
            except Exception:
                raise

        res = minimize_scalar(
            ess_err,
            method="brent",
            bounds=(alpha_lower, alpha_upper),
            options={"maxiter": 5000},
        )
        alpha_ = res.x
        log_w = -alpha_ * costs_
        ess, kl = policy.weighted_update(log_w, samples)
        self.trace["ess"].append(ess)
        self.trace["kl"].append(kl)
        self.trace["ent"].append(policy.entropy)
        self.trace["alpha"].append(alpha_)
        log_nw = log_w - logsumexp(log_w)
        self.trace["weight_ent"] += [(log_nw * np.exp(log_nw)).sum()]
        return policy
