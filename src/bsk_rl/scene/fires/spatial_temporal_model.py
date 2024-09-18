import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rxr  # Add to install
import scipy.integrate as integrate
import xarray as xr
from scipy.stats.sampling import NumericalInversePolynomial

from bsk_rl.scene.fires.exp_covariance_proc import ExponentialCovarianceProcess

time_model_params = dict(
    mu=2972.9140999315537,  # fires
    T=365.25,  # days
    # freqs=[0.0027378507871321013, 0.0054757015742642025, 0.008213552361396304],  # 1/day
    freqs=[1, 2, 3],  # per year
    mags=[796.2188699592335, 570.1721689939742, 318.7274992819098],  # fires
    phases=[2.109235880984748, -1.1881660156161062, 0.618303420964733],  # rad
    sigma=945.7936963761894,
    l=2.93497586385308,
    min_func=lambda x: 500 * np.log(1 + np.exp(x / 500)),
)


def fourier_generator(
    T: float, freqs: list[int], mags: list[float], phases: list[float], mu: float = 0.0
):
    """Generate a Fourier series function.

    Args:
        T: [t] Period of the function.
        freqs: [1/T] List of frequencies.
        mags: List of magnitudes.
        phases: [rad] List of phases.
        mu: Mean value.
    """

    def fourier(t):
        result = 0
        for freq, mag, phase in zip(freqs, mags, phases):
            result += mag * np.cos(freq * 2 * np.pi * t / T + phase)
        return result + mu

    return fourier


def load_fire_distribution():
    fire_distribution = pickle.load(
        open(
            Path(__file__).resolve().parent / "_dat" / "fire_count_distribution.pkl",
            "rb",
        )
    )
    prepend = fire_distribution[-1].copy()
    prepend["time"] -= pd.Timedelta(365.25, "d")
    append = fire_distribution[0].copy()
    append["time"] += pd.Timedelta(365.25, "d")
    fire_distribution = xr.concat([prepend, fire_distribution, append], dim="time")
    return fire_distribution


class FireLocationGenerator:
    def __init__(self, initial_day: float, final_day: float):
        self.initial_day = initial_day
        self.final_day = final_day
        self.duration = final_day - initial_day
        self.time_model_params = time_model_params

        self.intensity_fn = self.intensity_fn_generator(**self.time_model_params)
        self.fire_times = self.generate_fire_times()

        self.tile_size = 0.25
        self.fire_distribution = load_fire_distribution()
        self.fire_locations = self.generate_fire_locations()

    def intensity_fn_generator(
        self, T, mu, freqs, mags, phases, sigma, l, min_func=None, truncation=0.99
    ):
        """Generate a time process for an event occurrence.

        Results in a function that returns the sum of a Fourier series and an exponential
        covariance process.

        Args:
            T: [t] Period of the function.
            mu: Mean value.
            freqs: [1/T] List of frequencies.
            mags: List of magnitudes.
            phases: [rad] List of phases.
            sigma: Standard deviation of exponential process.
            l: [t] Length scale of exponential process.
            periods: Number of periods to generate.
            truncation: Truncation value for the exponential covariance process.
            min_func: Minimum function for the process.
        """
        fourier = fourier_generator(T, freqs, mags, phases, mu)
        exp_proc = ExponentialCovarianceProcess(
            mu=0, sigma=sigma, l=l, x_low=self.initial_day, x_high=self.final_day
        )
        realization = exp_proc.kle_realization(alpha=truncation)

        if min_func is None:
            min_func = lambda t: np.zeros_like(t)

        def intensity(t):
            result = realization(t) + fourier(t)
            return np.maximum(min_func(result), result)

        return intensity

    def generate_fire_times(self):
        initial_day = self.initial_day
        final_day = self.initial_day + self.duration

        volume = integrate.quad(
            self.intensity_fn, initial_day, final_day, epsabs=1.0, epsrel=0.001
        )[0]
        self.fire_count = np.random.poisson(volume)
        f = lambda t: self.intensity_fn(t) / volume

        class Process:
            def pdf(self, t):
                return f(t)

            def support(self):
                return (initial_day, final_day)

            def volume(self):
                return volume

        sampler = NumericalInversePolynomial(
            Process(),
            u_resolution=1e-5,
            order=3,
            center=(initial_day + final_day) / 2,
            domain=(initial_day, final_day),
        )

        time_samples = sampler.rvs(self.fire_count)
        time_samples.sort()

        return time_samples

    def generate_fire_locations(self):
        fire_locations = []
        for day in np.unique(np.round(self.fire_times)):
            daily_count = np.sum(np.round(self.fire_times) == day)
            fire_distribution_daily = self.fire_distribution.interp(
                dict(
                    x=self.fire_distribution.x,
                    y=self.fire_distribution.y,
                    time=pd.date_range(
                        start="1900-01-01", end="1900-01-01", freq=f"1D"
                    )[0]
                    + pd.Timedelta(day % 365, "d"),
                )
            )
            lats, longs = np.meshgrid(
                fire_distribution_daily.x, fire_distribution_daily.y
            )
            possible_locations = np.array(list(zip(lats.flatten(), longs.flatten())))
            land_area_sum = np.sum(fire_distribution_daily.data)

            random_indices = np.random.choice(
                np.arange(len(possible_locations)),
                size=daily_count,
                p=np.abs(fire_distribution_daily.data.flatten() / land_area_sum),
            )
            fire_locations.extend(possible_locations[random_indices])

        fire_locations = np.array(fire_locations)
        fire_locations += np.random.uniform(
            -self.tile_size / 2, self.tile_size / 2, size=fire_locations.shape
        )

        return fire_locations

    def __getitem__(self, key):
        return self.fire_times[key], self.fire_locations[key]

    def __len__(self):
        return self.fire_count


if __name__ == "__main__":
    import itertools

    import matplotlib.pyplot as plt

    from bsk_rl.utils.orbital import lla2ecef

    # dists = []
    dist_map = {1: [], 2: [], 5: [], 10: []}
    f = FireLocationGenerator(initial_day=10, final_day=12)

    locs = np.array([lla2ecef(fi[1][1], fi[1][0], 6371) for fi in f])
    for loc in locs:
        diff = np.linalg.norm(locs - loc, ord=2, axis=1)
        diff.sort()
        # dists.append(diff[1])
        for key in dist_map:
            dist_map[key].append(diff[key])

    fig, ax = plt.subplots()
    for key, dists in dist_map.items():
        ax.hist(
            dists,
            bins=np.arange(0, max(dists), 5),
            alpha=0.5,
            label=f"{key}th nearest",
            density=True,
        )
    # ax.hist(dists, bins=np.arange(0, max(dists), 5))
    ax.legend()
    plt.show()
