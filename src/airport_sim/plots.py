"""Standalone report figures using Matplotlib's noninteractive backend."""
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def save(fig, output, name):
    fig.tight_layout()
    fig.savefig(Path(output) / f"{name}.png", dpi=180)
    fig.savefig(Path(output) / f"{name}.pdf")
    plt.close(fig)


def mm1_plot(rows, output):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rho = [r["rho"] for r in rows]
    ax.plot(rho, [r["theory"] for r in rows], "--", label="M/M/1 theory")
    ax.errorbar(rho, [r["mean"] for r in rows],
                yerr=[r["ci_high"] - r["mean"] for r in rows], fmt="o", capsize=4,
                label="Simulation, 95% CI")
    ax.set(xlabel="Utilization (rho)", ylabel="Mean queue wait (minutes)", title="Steady-state validation; mean service = 1 min")
    ax.legend()
    save(fig, output, "mm1_validation")


def experiment_plots(summary, differences, cfg, output, synthetic):
    label = ("Calibrated synthetic inputs" if cfg.get("input_mode") == "calibrated_synthetic"
             else "Synthetic demonstration" if synthetic else "BTS-derived subset; assumed gate capacities")
    if cfg.get("demand_multiplier", 1) > 1:
        label += f"; {cfg['demand_multiplier']}x constructed demand"
    lookup = {(r["scenario"], r["metric"]): r for r in summary}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for variant, title in (("base", "Original variability"), ("low_variability", "Half the service CV")):
        rows = [lookup[f"single_{rho:g}_{variant}", "mean_wait"] for rho in cfg["single_loads"]]
        axes[0].errorbar(cfg["single_loads"], [r["mean"] for r in rows],
                         yerr=[r["ci_high"] - r["mean"] for r in rows], capsize=3, marker="o", label=title)
    axes[0].set(xlabel="Offered load (rho)", ylabel="Mean wait (minutes)", title="Single gate: finite day, empty start")
    axes[0].legend(fontsize=8)
    names = ["terminal_dedicated", "terminal_pooled", "terminal_smoothed"]
    rows = [lookup[name, "mean_wait"] for name in names]
    axes[1].bar(["Dedicated", "Pooled", "Smoothed"], [r["mean"] for r in rows],
                yerr=[r["ci_high"] - r["mean"] for r in rows], capsize=4, color=["#718096", "#2b6cb0", "#2c7a7b"])
    axes[1].set(ylabel="Mean wait (minutes)", title="Reference terminal")
    rows = [lookup[f"network_sharing_{x:g}", "mean_wait"] for x in cfg["transfer_times"]]
    axes[2].errorbar(cfg["transfer_times"], [r["mean"] for r in rows],
                     yerr=[r["ci_high"] - r["mean"] for r in rows], capsize=4, marker="o", label="Sharing (includes transfer)")
    baseline = lookup["network_no_sharing", "mean_wait"]
    axes[2].axhline(baseline["mean"], linestyle="--", label="No sharing")
    axes[2].axhspan(baseline["ci_low"], baseline["ci_high"], alpha=.12)
    axes[2].set(xlabel="Transfer time (minutes)", ylabel="Mean wait (minutes)", title="Multiple terminals")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{label} — replication means with 95% CIs", fontsize=12)
    save(fig, output, "experiment_overview")
    diff = [r for r in differences if r["metric"] == "mean_wait" and not r["alternative"].startswith("single_")]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar([r["mean"] for r in diff], range(len(diff)),
                xerr=[r["ci_high"] - r["mean"] for r in diff], fmt="o", capsize=4)
    ax.set_yticks(range(len(diff)), [r["alternative"].replace("_", " ") for r in diff])
    ax.axvline(0, linestyle="--", color="gray")
    ax.set(xlabel="Alternative − baseline mean wait (minutes); 95% paired CI",
           title=("Synthetic inputs" if synthetic else "BTS-derived inputs") + ": paired policy differences")
    save(fig, output, "policy_differences")


def input_diagnostics(values, fits, days, train_dates, holdout_dates, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(values, bins="auto", density=True, alpha=.4, label="Training observations")
    x = np.linspace(max(.001, min(values) * .5), max(values) * 1.1, 300)
    for fit in fits:
        distribution = getattr(stats, fit["distribution"])
        axes[0].plot(x, distribution.pdf(x, *fit["parameters"]), label=fit["distribution"])
    axes[0].set(xlabel="Service time (minutes)", ylabel="Density", title="Input distribution fits")
    axes[0].legend(fontsize=8)
    best = min(fits, key=lambda r: r["aic"])
    dist = getattr(stats, best["distribution"])
    probs = (np.arange(len(values)) + .5) / len(values)
    theoretical = dist.ppf(probs, *best["parameters"])
    axes[1].scatter(theoretical, np.sort(values), s=8)
    limit = max(max(values), max(theoretical))
    axes[1].plot([0, limit], [0, limit], "--", color="gray")
    axes[1].set(xlabel="Fitted quantile", ylabel="Observed quantile", title=f"Training Q-Q: {best['distribution']}")
    for dates, label in ((train_dates, "Training"), (holdout_dates, "Holdout")):
        if dates:
            counts = np.mean([np.histogram([f.arrival for f in days[d]], bins=np.arange(0, 1441, 30))[0] for d in dates], axis=0)
            axes[2].plot(np.arange(15, 1440, 30) / 60, counts, label=label)
    axes[2].set(xlabel="Hour of day", ylabel="Mean arrivals / 30 min", title="Arrival peaks")
    axes[2].legend()
    save(fig, output, "input_diagnostics")


def terminal_policy_plots(summary, scenario_map, cfg, output):
    """Separate local queues: one panel per terminal, identical policy design."""
    lookup = {(r['scenario'], r['metric']): r for r in summary}
    fig, axes = plt.subplots(1, len(scenario_map), figsize=(4.2*len(scenario_map), 4), squeeze=False, sharey=True)
    for ax, (terminal, names) in zip(axes[0], scenario_map.items()):
        rows = [lookup[names[p], 'mean_wait'] for p in ['dedicated','pooled','smoothed']]
        ax.bar(['Dedicated','Pooled','Smoothed'], [r['mean'] for r in rows],
               yerr=[r['ci_high']-r['mean'] for r in rows], capsize=3,
               color=['#718096','#2b6cb0','#2c7a7b'])
        ax.set(title=f"{terminal}: {cfg['gates_per_terminal']} gates", ylabel='Mean wait (minutes)')
    fig.suptitle(f"Separate single-terminal experiments; {rows[0]['n']} replications per policy, 95% CIs")
    save(fig, output, 'all_terminal_policies')
