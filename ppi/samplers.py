import numpy as np
from numpy.random import multivariate_normal as mvn
from scipy.special import erfinv
from scipy.stats import qmc

__all__ = ["QuasiMonteCarlo", "MonteCarlo", "CubatureQuadrature", "Particles"]


class BaseSampler(object):
    def __call__(self, mu, sigma, n_samples, standard_gaussian):
        raise NotImplementedError


class MonteCarlo(BaseSampler):
    def __init__(self, dimension):
        self.d = dimension

    def __call__(self, mu, sigma, n_samples, standard_gaussian=False):
        if standard_gaussian:
            return np.random.randn(n_samples, self.d)
        else:
            return mvn(mu, sigma, size=(n_samples,))


class QuasiMonteCarlo(BaseSampler):
    def __init__(self, dimension):
        self.sampler = qmc.Sobol(d=dimension, scramble=True)

    def __call__(self, mu, sigma, n_samples, shrinkage=0.9999, standard_gaussian=False):
        with np.errstate(invalid="ignore"):
            n_samples_b2 = 2 ** int(np.ceil(np.log(n_samples) / np.log(2)))
            # if sobol is -1 or 1, gaussian is inf and samples are NaN, so apply tiny 'shrinkage'
            sobol = shrinkage * self.sampler.random(n_samples_b2)[:n_samples]
            base_gaussian = np.sqrt(2) * erfinv(2 * sobol - 1)
            if standard_gaussian:
                return base_gaussian
            else:
                sqrt = np.linalg.cholesky(sigma)
                samples = mu[None, :] + base_gaussian @ sqrt.T
                return samples

    @staticmethod
    def covariance_scale(n):
        return n - 1


class CubatureQuadrature(BaseSampler):
    def __init__(self, dimension):
        self.d = dimension
        self.base_points = np.sqrt(self.d) * np.concatenate(
            (np.eye(dimension), -np.eye(dimension))
        )
        print(f"Using {self.base_points.shape} samples for cubature quadrature")

    def __call__(self, mu, sigma, n_samples, standard_gaussian=False):
        if standard_gaussian:
            return self.base_points
        else:
            sqrt = np.linalg.cholesky(sigma)
            return mu[None, :] + self.base_points @ sqrt.T

    @property
    def n_samples(self):
        return 2 * self.d


class Particles(object):

    _particles: np.ndarray
    has_particles = False

    def __init__(self, dimension):
        self._particles = None

    @property
    def particles(self):
        return self._particles

    @particles.setter
    def particles(self, particles):
        self._particles = particles
        self.has_particles = True

    def __call__(self, mu, sigma, n_samples, standard_gaussian=False):
        if standard_gaussian:
            d = mu.shape[0]
            samples = np.random.randn(n_samples, d)
        else:
            samples = mvn(mu, sigma, size=(n_samples,))
        samples = self.add_particles(samples)
        return samples

    def add_particles(self, samples):
        if self._particles is not None:
            d_n, d_t, _ = samples.shape
            n = min(self._particles.shape[0], d_n)
            samples[:n, ...] = self._particles[:n, :d_t, :]
        return samples

    @staticmethod
    def covariance_scale(n):
        return n - 1
