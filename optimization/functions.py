"""
Test functions for optimization.
Based on https://github.com/hanyas/reps/blob/master/reps/envs/episodic/benchmarks.py
"""
from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Himmelblau", "Rosenbrock", "Rastrigin", "Styblinski", "NoisySphere"]


class Base(ABC):
    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Himmelblau(Base):

    f_opt = 0.0

    def __call__(self, x):
        a = x[:, 0] * x[:, 0] + x[:, 1] - 11.0
        b = x[:, 0] + x[:, 1] * x[:, 1] - 7.0
        return -1.0 * (a * a + b * b) - self.f_opt


class Rosenbrock(Base):

    f_opt = 0.0

    @property
    def x_opt(self):
        return np.zeros((self.dim,))

    def __call__(self, x):
        return (
            np.sum(
                100.0 * (x[:, 1:] - x[:, :-1] ** 2.0) ** 2 + (1.0 - x[:, :-1]) ** 2,
                axis=-1,
            )
            - self.f_opt
        )


class Styblinski(Base):
    """
    See
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    https://www.sfu.ca/~ssurjano/stybtang.html
    for details.
    """

    @property
    def x_opt(self):
        return -2.903534 * np.ones((self.dim,))

    @property
    def f_opt(self):
        return -39.16599 * self.dim

    def __call__(self, x):
        return 0.5 * np.sum(x ** 4.0 - 16.0 * x ** 2 + 5.0 * x, axis=-1) - self.f_opt


class Rastrigin(Base):

    f_opt = 0.0
    A = 10.0

    @property
    def x_opt(self):
        return np.zeros((self.dim,))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (
            self.A * self.dim
            + np.sum(x ** 2 - self.A * np.cos(2.0 * np.pi * x), axis=-1)
            - self.f_opt
        )


class NoisySphere(Base):
    """Randomized quadratic test function with evaluation noise."""

    sigma = 0.01
    f_opt = 0.0

    def __init__(self, dim: int, seed=0):
        super().__init__(dim)
        rng = np.random.default_rng(seed)
        chol = rng.standard_normal((dim, dim))
        self.A = chol @ chol.T

    @property
    def x_opt(self):
        return np.zeros((self.dim,))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        noise = self.sigma * np.random.randn(x.shape[0])
        return np.einsum("bi, ij, bj->b", x, self.A, x) + noise - self.f_opt
