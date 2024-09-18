from dataclasses import dataclass
from functools import cache
from typing import Optional

import numpy as np
from scipy.optimize import root_scalar
from tqdm import tqdm


@dataclass
class ExponentialCovarianceProcess:
    mu: float
    sigma: float
    l: float
    x_low: float
    x_high: float

    def __post_init__(self):
        self.a = (self.x_high - self.x_low) / 2
        self.offset = (self.x_high + self.x_low) / 2

    def __call__(self, t1, t2):
        return self.sigma**2 * np.exp(-np.abs(t1 - t2) / self.l)

    def __hash__(self):
        return hash((self.mu, self.sigma, self.l, self.a))

    def omega(self, i: int):
        epsilon = 1e-12
        if i % 2 == 1:

            def rootfn(omega):
                return 1 / self.l - omega * np.tan(omega * self.a)

            interval = [
                max(0, (i - 2) * np.pi / (2 * self.a) + epsilon),
                (i) * np.pi / (2 * self.a) - epsilon,
            ]
        else:

            def rootfn(omega):
                # print(omega, omega + (1 / self.l) * np.tan(omega * self.a))
                return omega + (1 / self.l) * np.tan(omega * self.a)

            interval = [
                max(epsilon, (i - 1) * np.pi / (2 * self.a) + epsilon),
                (i + 1) * np.pi / (2 * self.a) - epsilon,
            ]
            # print(interval)
        omega_i = root_scalar(rootfn, x0=0, bracket=interval).root
        return omega_i

    @cache
    def eigenvalue(self, i: int):
        return 2 * self.l * self.sigma**2 / (self.l**2 * self.omega(i) ** 2 + 1)

    @cache
    def eigenfunction(self, i: int):
        omega = self.omega(i)
        if i % 2 == 1:
            return lambda t: np.cos(omega * (t - self.offset)) / np.sqrt(
                self.a + np.sin(2 * omega * self.a) / (2 * omega)
            )
        else:
            return lambda t: np.sin(omega * (t - self.offset)) / np.sqrt(
                self.a - np.sin(2 * omega * self.a) / (2 * omega)
            )

    def truncation_msn(self, j: int, k: int = 20):
        eigenvalues = [self.eigenvalue(i) for i in range(1, j + k + 1)]
        return sum(eigenvalues[0:-k]) / sum(eigenvalues)

    def terms_for_truncation(self, alpha=0.9):
        i = 1
        while self.truncation_msn(i) < alpha:
            i += 1
        return i

    def kle_realization(self, d: Optional[int] = None, alpha: Optional[float] = None):
        if d is not None and alpha is not None:
            raise ValueError("Only one of d and alpha should be provided.")

        if d is None and alpha is None:
            alpha = 0.9

        if alpha is not None:
            d = self.terms_for_truncation(alpha)

        eigenfunctions = [self.eigenfunction(i) for i in range(1, d + 1)]
        eigenvalues = [self.eigenvalue(i) for i in range(1, d + 1)]
        ys = np.random.normal(size=d)

        def kle(t):
            return (
                sum(
                    np.sqrt(eigenvalues[i]) * ys[i] * eigenfunctions[i](t)
                    for i in range(d)
                )
                + self.mu
            )

        return kle
