"""Ch. 3 reproducible streams; Ch. 4 inverse transform and convolution."""
import numpy as np


def streams(seed, replication):
    """Separate day/arrival, service, and routing streams for each replication."""
    return tuple(np.random.default_rng(s) for s in
                 np.random.SeedSequence([seed, replication]).spawn(3))


def exponential(rng, mean, size=None):
    if mean <= 0:
        raise ValueError("Exponential mean must be positive")
    return -mean * np.log1p(-rng.random(size))


def erlang(rng, mean, shape, size):
    """Convolve k exponentials: mean unchanged, CV = 1/sqrt(k)."""
    if not isinstance(shape, int) or shape < 1:
        raise ValueError("Erlang shape must be a positive integer")
    return exponential(rng, mean / shape, (size, shape)).sum(axis=1)


def poisson_arrivals(rng, rate, horizon):
    if rate <= 0 or horizon <= 0:
        raise ValueError("Rate and horizon must be positive")
    times, t = [], 0.0
    while True:
        t += float(exponential(rng, 1 / rate))
        if t >= horizon:
            return np.asarray(times)
        times.append(t)
