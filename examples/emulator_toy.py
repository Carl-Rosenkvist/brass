"""Try brass.emulation.Emulator on made-up data (no SMASH runs needed).

    python examples/emulator_toy.py

A toy "model" gives dN/dy (y >= 0, 15 bins) of a "pion" and a "proton" as
a function of the collision energy and two model parameters a and b. Each
training run gets Poisson noise, as from a finite number of events. The
script

1. fits the emulator on 100 parameter points, each at two energies,
2. compares it with the noise-free model at new points,
3. checks its uncertainty on held-out runs (z = (emulator - truth) / sigma),
4. recovers known parameters from fake data with emcee,

and writes emulator_toy.png.
"""

import time

import emcee
import matplotlib.pyplot as plt
import numpy as np
from brass.emulation import Emulator, EmulatorLikelihood

ENERGIES = (8.76, 17.3)
NAMES = ["Sqrtsnn", "a", "b"]
BOUNDS = {"a": (0.5, 1.5), "b": (0.6, 1.4)}
Y_EDGES = np.linspace(0.0, 3.0, 16)
Y = 0.5 * (Y_EDGES[1:] + Y_EDGES[:-1])
EVENTS = 5000  # events per training run
TRUTH = {"a": 1.1, "b": 0.85}  # parameters of the fake data


def model(sqrts, a, b):
    """Noise-free dN/dy, {"pion": (n, 15), "proton": (n, 15)}."""
    sqrts, a, b = (
        np.atleast_1d(np.asarray(v, dtype=float))[:, None] for v in (sqrts, a, b)
    )
    log_s = np.log(sqrts)
    pion = a * log_s * np.exp(-0.5 * (Y / (b * (0.5 + 0.25 * log_s))) ** 2)
    peak = 0.3 * log_s * b  # the protons' humps move out with energy and b
    proton = (0.3 + 0.2 * a) * (
        np.exp(-0.5 * ((Y - peak) / 0.5) ** 2) + np.exp(-0.5 * ((Y + peak) / 0.5) ** 2)
    )
    return {"pion": pion, "proton": proton}


def simulate(sqrts, a, b, rng):
    """Model with the statistical noise of EVENTS events per run."""
    dy = np.diff(Y_EDGES)
    return {
        key: rng.poisson(exact * dy * EVENTS) / (dy * EVENTS)
        for key, exact in model(sqrts, a, b).items()
    }


def main():
    rng = np.random.default_rng(1)

    # -- 1. training runs and fit ---------------------------------------------
    n_points = 100
    a = rng.uniform(*BOUNDS["a"], n_points)
    b = rng.uniform(*BOUNDS["b"], n_points)
    X = np.array([[energy, ai, bi] for energy in ENERGIES for ai, bi in zip(a, b)])
    training = simulate(X[:, 0], X[:, 1], X[:, 2], rng)

    emu = Emulator(X, names=NAMES)
    for key, values in training.items():
        emu.add(key, values, bins=Y_EDGES)
    start = time.perf_counter()
    emu.fit(verbose=True)
    print(f"fit: {time.perf_counter() - start:.1f} s for {len(X)} runs\n")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

    # -- 2. new points: emulator against the noise-free model -------------------
    new = {"a": rng.uniform(*BOUNDS["a"], 3), "b": rng.uniform(*BOUNDS["b"], 3)}
    theta = {"Sqrtsnn": 17.3, **new}
    exact = model(17.3, new["a"], new["b"])
    for ax, key in zip(axes[0, :2], ("pion", "proton")):
        mean, std = emu.predict(theta, key, return_std=True)
        for i, color in enumerate(("C0", "C1", "C2")):
            ax.fill_between(
                Y, mean[i] - 2 * std[i], mean[i] + 2 * std[i], color=color, alpha=0.3
            )
            ax.plot(
                Y,
                mean[i],
                color=color,
                label=f"a={new['a'][i]:.2f}, b={new['b'][i]:.2f}",
            )
            ax.plot(Y, exact[key][i], "k--", lw=0.8)
        ax.set(
            title=f"{key}, 17.3 GeV: emulator ±2σ, model dashed",
            xlabel="y",
            ylabel="dN/dy",
        )
        ax.legend(fontsize=8)

    # -- 3. held-out runs: is the uncertainty right? ----------------------------
    results = emu.validate(holdout=0.2)
    z_runs, z_model = [], []
    for key, r in results.items():
        p = r["parameters"]
        truth = model(p[:, 0], p[:, 1], p[:, 2])[key]
        z_runs.append(r["residuals"].ravel())  # against the noisy held-out runs
        z_model.append(((r["mean"] - truth) / r["std"]).ravel())  # against the model
    ax = axes[0, 2]
    x = np.linspace(-4, 4, 200)
    for z, label in ((z_model, "vs noise-free model"), (z_runs, "vs noisy runs")):
        z = np.concatenate(z)
        z = z[np.isfinite(z)]
        print(
            f"held-out z {label}: std {z.std():.2f}, |z|<1: {np.mean(np.abs(z) < 1):.0%}, "
            f"|z|<2: {np.mean(np.abs(z) < 2):.0%}"
        )
        ax.hist(
            z,
            bins=np.linspace(-4, 4, 41),
            density=True,
            histtype="step",
            label=f"{label} (std {z.std():.2f})",
        )
    ax.plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), "k--", label="N(0, 1)")
    ax.set(title="held-out runs", xlabel="z = (emulator - truth) / σ")
    ax.legend(fontsize=8)

    # -- 4. calibration: recover TRUTH from fake data ---------------------------
    like = EmulatorLikelihood(emu)
    for energy in ENERGIES:
        for key, values in model(energy, TRUTH["a"], TRUTH["b"]).items():
            err = 0.05 * values[0] + 1e-3
            data = values[0] + err * rng.standard_normal(len(Y))
            like.add(key, Y, data, err, fixed={"Sqrtsnn": energy})
    print(f"\nfitted parameters: {like.parameters}, {like.n_points} data points")

    lower = np.array([BOUNDS[n][0] for n in like.parameters])
    upper = np.array([BOUNDS[n][1] for n in like.parameters])

    def log_probability(thetas):
        logp = np.full(len(thetas), -np.inf)
        inside = np.all((thetas >= lower) & (thetas <= upper), axis=1)
        if inside.any():
            logp[inside] = like(thetas[inside])
        return logp

    n_walkers = 16
    sampler = emcee.EnsembleSampler(
        n_walkers, len(lower), log_probability, vectorize=True
    )
    start = time.perf_counter()
    state = sampler.run_mcmc(rng.uniform(lower, upper, (n_walkers, len(lower))), 500)
    sampler.reset()
    sampler.run_mcmc(state, 2000)
    samples = sampler.get_chain(flat=True)
    print(
        f"MCMC: {time.perf_counter() - start:.1f} s for 2500 steps x {n_walkers} walkers"
    )
    for i, name in enumerate(like.parameters):
        lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
        print(f"  {name}: {med:.3f} [{lo:.3f}, {hi:.3f}]   true {TRUTH[name]}")

    for i, name in enumerate(like.parameters):
        ax = axes[1, i]
        ax.hist(samples[:, i], bins=50, color="0.6")
        ax.axvline(TRUTH[name], color="C3", label="true value")
        ax.set(title=f"posterior of {name}", xlabel=name)
        ax.legend(fontsize=8)
    ax = axes[1, 2]
    ax.plot(samples[::10, 0], samples[::10, 1], ".", ms=1, color="0.5")
    ax.plot(TRUTH["a"], TRUTH["b"], "*", color="C3", ms=12)
    ax.set(title="posterior samples", xlabel="a", ylabel="b")

    fig.tight_layout()
    fig.savefig("emulator_toy.png", dpi=120)
    print("\nwrote emulator_toy.png")


if __name__ == "__main__":
    main()
