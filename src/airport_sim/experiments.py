"""Experiment runner: common inputs, independent replications, explicit provenance."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import platform
import hashlib
import importlib.metadata
import numpy as np
from scipy.stats import gamma
from .inputs import load_days, synthetic_day, write_csv
from .metrics import confidence_interval, paired_difference
from .random_inputs import exponential, poisson_arrivals, streams
from .simulator import Flight, Gate, simulate


DEFAULT_CONFIG = dict(horizon=1440, flights_per_day=200, mean_service=45,
                      gates_per_terminal=4, shared_gates_per_terminal=2,
                      transfer_times=[0, 10, 20, 40], single_loads=[.5, .7, .85, .95],
                      closure_start=480, closure_duration=120, smoothing=.5,
                      initial_busy_until=0, input_mode="empirical", service_mode="resample_by_airline",
                      demand_multiplier=1)


def configuration(path=None):
    cfg = DEFAULT_CONFIG.copy()
    if path:
        given = json.loads(Path(path).read_text())
        unknown = set(given) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        cfg.update(given)
    if cfg["input_mode"] not in {"empirical", "calibrated_synthetic"}:
        raise ValueError("Invalid input_mode")
    if cfg["input_mode"] == "calibrated_synthetic" and cfg["service_mode"] == "observed_day":
        raise ValueError("Synthetic inputs cannot replay observed services")
    if cfg["service_mode"] not in {"resample_by_airline", "observed_day"}:
        raise ValueError("service_mode must be resample_by_airline or observed_day")
    for key in ("horizon", "mean_service"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("gates_per_terminal", "flights_per_day", "demand_multiplier"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(cfg["shared_gates_per_terminal"], int) or not 0 <= cfg["shared_gates_per_terminal"] <= cfg["gates_per_terminal"]:
        raise ValueError("Invalid shared gate count")
    if not cfg["single_loads"] or any(not 0 < rho < 1 for rho in cfg["single_loads"]):
        raise ValueError("Single gate loads must lie strictly between zero and one")
    if len(set(cfg["single_loads"])) != len(cfg["single_loads"]):
        raise ValueError("Single gate loads must be unique")
    if not cfg["transfer_times"] or any(not np.isfinite(x) or x < 0 for x in cfg["transfer_times"]):
        raise ValueError("Transfer times must be finite and nonnegative")
    if len(set(cfg["transfer_times"])) != len(cfg["transfer_times"]):
        raise ValueError("Transfer times must be unique")
    for key in ("closure_start", "closure_duration", "initial_busy_until"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    if cfg["closure_duration"] == 0 or not 0 <= cfg["smoothing"] <= 1:
        raise ValueError("Closure duration must be positive and smoothing in [0,1]")
    return cfg


def make_gates(airlines_by_terminal, cfg, initial=None):
    all_airlines = tuple(sorted({a for v in airlines_by_terminal.values() for a in v}))
    gates = []
    for terminal, airlines in sorted(airlines_by_terminal.items()):
        for index in range(cfg["gates_per_terminal"]):
            compatible = all_airlines if index < cfg["shared_gates_per_terminal"] else tuple(sorted(airlines))
            gates.append(Gate(f"{terminal}-G{index + 1}", terminal, compatible,
                              initial_busy_until=cfg["initial_busy_until"] if initial is None else initial))
    return gates


def smooth_day(flights, horizon, strength):
    ordered = sorted(flights, key=lambda f: (f.arrival, f.id))
    return [replace(f, arrival=(1 - strength) * f.arrival + strength * (i + .5) * horizon / len(ordered))
            for i, f in enumerate(ordered)]


def expand_day(flights, multiplier, rng, horizon):
    """Keep originals; add unique flights within each source arrival's 30-min bin.

    This is a constructed demand stress test, never additional observed traffic.
    Copy-major order makes lower demand levels prefixes of higher levels.
    """
    expanded = list(flights)
    used_ids = {f.id for f in flights}
    for copy in range(1, multiplier):
        for f in flights:
            identifier = f"{f.id}::stress{copy}"
            while identifier in used_ids:
                identifier += ":"
            used_ids.add(identifier)
            lower = 30 * np.floor(f.arrival / 30)
            expanded.append(replace(f, id=identifier,
                arrival=float(rng.uniform(lower, min(lower + 30, horizon)))))
    return expanded


def run_experiments(output, replications=100, seed=553, config_path=None, data_path=None):
    if replications < 2 or seed < 0:
        raise ValueError("Use at least two replications and a nonnegative seed")
    cfg = configuration(config_path)
    horizon = cfg["horizon"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    days = load_days(data_path, horizon) if data_path else None
    calibrated = cfg["input_mode"] == "calibrated_synthetic"
    if calibrated and not days:
        raise ValueError("Calibrated synthetic inputs require --data")
    if cfg["demand_multiplier"] != 1 and not days:
        raise ValueError("Demand stress tests require a reference --data file")
    all_flights = [f for fs in days.values() for f in fs] if days else []
    airlines_by_terminal = {}
    for f in all_flights:
        airlines_by_terminal.setdefault(f.terminal, set()).add(f.airline)
    if not days:
        airlines_by_terminal = {"T1": {"A"}, "T2": {"B"}}
    gates = make_gates(airlines_by_terminal, cfg)
    selected_terminal = sorted(airlines_by_terminal)[0]
    local_gates = [g for g in gates if g.terminal == selected_terminal]
    terminal_scenarios = {
        terminal: {policy: (f"terminal_{policy}" if terminal == selected_terminal
                            else f"terminal_{terminal}_{policy}")
                   for policy in ("dedicated", "pooled", "smoothed", "pooled_independent")}
        for terminal in sorted(airlines_by_terminal)}
    service_pools = {a: [f.service for f in all_flights if f.airline == a]
                     for a in {f.airline for f in all_flights}}

    profiles = {}
    for airline, pool in service_pools.items():
        mean, variance = float(np.mean(pool)), float(np.var(pool))
        bins = np.linspace(0, horizon, int(np.ceil(horizon / 30)) + 1)
        counts, _ = np.histogram([f.arrival for f in all_flights if f.airline == airline], bins)
        profiles[airline] = dict(mean=mean, variance=variance,
                                 bin_edges=bins.tolist(), probabilities=(counts / counts.sum()).tolist())

    def gamma_services(uniforms, pool):
        mean, variance = float(np.mean(pool)), float(np.var(pool))
        return (gamma.ppf(np.clip(uniforms, np.finfo(float).eps, 1 - np.finfo(float).eps),
                          mean * mean / variance, scale=variance / mean)
                if variance else np.full(len(uniforms), mean))

    def day(rep, alternate_seed=seed):
        if not days:
            return synthetic_day(alternate_seed, rep, horizon, cfg["mean_service"], cfg["flights_per_day"]), "SYNTHETIC"
        arrival_rng, service_rng, _ = streams(alternate_seed, rep)
        selected = list(days)[int(arrival_rng.integers(len(days)))]
        # Separate stream: expansion does not alter date or service draws.
        expansion_rng, _, _ = streams(alternate_seed + 202, rep)
        templates = expand_day(days[selected], cfg["demand_multiplier"], expansion_rng, horizon)
        if calibrated:
            generated = []
            for f in templates:
                profile = profiles[f.airline]
                index = int(arrival_rng.choice(len(profile["probabilities"]), p=profile["probabilities"]))
                edges = profile["bin_edges"]
                arrival = float(arrival_rng.uniform(edges[index], edges[index + 1]))
                service = float(gamma_services(service_rng.random(1), service_pools[f.airline])[0])
                generated.append(replace(f, arrival=arrival, service=service))
            return generated, selected
        if cfg["service_mode"] == "observed_day":
            return templates, selected
        # Bind the service draw to the flight before any policy executes.
        return [replace(f, service=float(service_rng.choice(service_pools[f.airline])))
                for f in templates], selected

    rows, comparisons = [], []
    single_mean = float(np.mean([f.service for f in all_flights])) if days else cfg["mean_service"]
    single_pool = np.sort([f.service for f in all_flights]) if days else None
    for rep in range(replications):
        flights, selected_date = day(rep)
        def record(name, demand, capacity, policy="pooled", transfer=0, adjustment=0, input_date=None):
            result = simulate(demand, capacity, policy, horizon, transfer, trace=rep == 0 and name == "network_no_sharing")
            rows.append(dict(replication=rep, scenario=name, sampled_date=selected_date if input_date is None else input_date,
                             mean_schedule_adjustment=adjustment, **result.metrics))
            if rep == 0:
                write_csv(output / "flight_records" / f"{name}.csv", result.flights)
                if result.trace:
                    (output / "event_trace.json").write_text(json.dumps(result.trace, indent=2))
            return result

        local = [f for f in flights if f.terminal == selected_terminal]
        record("terminal_dedicated", local, local_gates, "dedicated")
        record("terminal_pooled", local, local_gates)
        smooth = smooth_day(local, horizon, cfg["smoothing"])
        old_times = {f.id: f.arrival for f in local}
        adjustment = float(np.mean([abs(f.arrival - old_times[f.id]) for f in smooth])) if smooth else 0
        record("terminal_smoothed", smooth, local_gates, adjustment=adjustment)
        record("network_no_sharing", flights, gates)
        for transfer in cfg["transfer_times"]:
            record(f"network_sharing_{transfer:g}", flights, gates, "sharing", transfer)
        closed = [replace(g, closures=((cfg["closure_start"], cfg["closure_start"] + cfg["closure_duration"]),))
                  if i == 0 else g for i, g in enumerate(gates)]
        record("network_closed_gate", flights, closed)
        record("network_initial_occupied", flights, make_gates(airlines_by_terminal, cfg, single_mean / 2))
        record("network_initial_empty", flights, make_gates(airlines_by_terminal, cfg, 0))

        independent_day, _ = day(rep, seed + 1000003)
        independent = simulate([f for f in independent_day if f.terminal == selected_terminal], local_gates, horizon=horizon)
        rows.append(dict(replication=rep, scenario="terminal_pooled_independent", sampled_date="INDEPENDENT_DRAW",
                         mean_schedule_adjustment=0.0, **independent.metrics))

        for terminal, names in terminal_scenarios.items():
            if terminal == selected_terminal:
                continue  # Keep the original reference-terminal outputs unchanged.
            demand = [f for f in flights if f.terminal == terminal]
            capacity = [g for g in gates if g.terminal == terminal]
            record(names["dedicated"], demand, capacity, "dedicated")
            record(names["pooled"], demand, capacity)
            smoothed = smooth_day(demand, horizon, cfg["smoothing"])
            old = {f.id: f.arrival for f in demand}
            adjustment = float(np.mean([abs(f.arrival-old[f.id]) for f in smoothed])) if smoothed else 0
            record(names["smoothed"], smoothed, capacity, adjustment=adjustment)
            record(names["pooled_independent"],
                   [f for f in independent_day if f.terminal == terminal], capacity,
                   input_date="INDEPENDENT_DRAW")

        for rho in cfg["single_loads"]:
            arrival_rng, service_rng, _ = streams(seed + 101, rep)
            times = poisson_arrivals(arrival_rng, rho / single_mean, horizon)
            u = service_rng.random(len(times))
            if calibrated:
                services = gamma_services(u, single_pool)
            elif days:
                services = single_pool[np.minimum((u * len(single_pool)).astype(int), len(single_pool) - 1)]
            else:
                services = -single_mean * np.log1p(-u)
            # Common-input contraction preserves population mean and reduces CV.
            reduced = single_mean + .5 * (services - single_mean)
            capacity = [Gate("S1", "T1")]
            for variant, duration in (("base", services), ("low_variability", reduced)):
                demand = [Flight(f"S{i:06d}", float(a), float(s)) for i, (a, s) in enumerate(zip(times, duration))]
                record(f"single_{rho:g}_{variant}", demand, capacity, input_date="SYNTHETIC_POISSON_ARRIVALS")
        print(f"Experiment replication {rep + 1}/{replications}", flush=True)

    scenario_names = sorted({r["scenario"] for r in rows})
    metric_names = [k for k in rows[0] if k not in {"replication", "scenario", "sampled_date"}]
    def values(name, metric):
        return [r[metric] for r in rows if r["scenario"] == name]
    summary = [dict(scenario=name, metric=metric, **confidence_interval(values(name, metric)))
               for name in scenario_names for metric in metric_names]
    pairs = [("terminal_dedicated", "terminal_pooled"), ("terminal_pooled", "terminal_smoothed"),
             ("network_no_sharing", "network_closed_gate"),
             ("network_initial_empty", "network_initial_occupied")]
    pairs += [("network_no_sharing", f"network_sharing_{x:g}") for x in cfg["transfer_times"]]
    pairs += [(f"single_{rho:g}_base", f"single_{rho:g}_low_variability") for rho in cfg["single_loads"]]
    for terminal, names in terminal_scenarios.items():
        if terminal != selected_terminal:
            pairs += [(names["dedicated"], names["pooled"]), (names["pooled"], names["smoothed"])]
    for baseline, alternative in pairs:
        for metric in metric_names:
            comparisons.append(dict(baseline=baseline, alternative=alternative, metric=metric,
                                    **paired_difference(values(baseline, metric), values(alternative, metric))))
    paired = np.asarray(values("terminal_pooled", "mean_wait")) - np.asarray(values("terminal_dedicated", "mean_wait"))
    independent = np.asarray(values("terminal_pooled_independent", "mean_wait")) - np.asarray(values("terminal_dedicated", "mean_wait"))
    paired_var, independent_var = float(paired.var(ddof=1)), float(independent.var(ddof=1))
    crn = dict(paired_difference_variance=paired_var, independent_difference_variance=independent_var,
               variance_ratio_independent_over_crn=independent_var / paired_var if paired_var else None,
               note="Empirical variance comparison; CRN is not guaranteed to improve every scenario.")
    crn_by_terminal = {}
    for terminal, names in terminal_scenarios.items():
        baseline = np.asarray(values(names["dedicated"], "mean_wait"))
        pv = float((np.asarray(values(names["pooled"], "mean_wait"))-baseline).var(ddof=1))
        iv = float((np.asarray(values(names["pooled_independent"], "mean_wait"))-baseline).var(ddof=1))
        crn_by_terminal[terminal] = dict(paired_difference_variance=pv,
            independent_difference_variance=iv, variance_ratio_independent_over_crn=iv/pv if pv else None)
    if not days:
        service_sampling = "Synthetic Erlang(4), single-gate exponential"
    elif calibrated:
        service_sampling = "Moment-matched Gamma; empirical half-hour arrival profile"
    elif cfg["service_mode"] == "observed_day":
        service_sampling = ("Source-service copies with generated extra arrivals; not an observed-day replay"
                            if cfg["demand_multiplier"] > 1
                            else "Observed day services (dependence-preserving replay)")
    else:
        service_sampling = "By-airline empirical resampling"
    metadata = dict(seed=seed, replications=replications,
                    config={k: v for k, v in cfg.items() if not days or k not in {"flights_per_day", "mean_service"}}, python=platform.python_version(),
                    packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")},
                    input_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest() if days else None,
                    input_source=str(Path(data_path).resolve()) if days else "SYNTHETIC DEMONSTRATION — no BTS calibration",
                    selected_terminal=selected_terminal, gates=[asdict(g) for g in gates],
                    terminal_scenarios=terminal_scenarios, crn_by_terminal=crn_by_terminal,
                    input_dates=list(days) if days else [],
                    calibration_profiles=profiles if calibrated else None,
                    input_mode=cfg["input_mode"] if days else "legacy_synthetic",
                    demand_expansion=dict(multiplier=cfg["demand_multiplier"],
                        applies_to="Terminal and network only; single-gate offered loads unchanged",
                        interpretation="Constructed BTS-based stress test" if cfg["demand_multiplier"] > 1 else "Reference demand",
                        method="Keep originals; added empirical arrivals uniform within source 30-minute bin; synthetic arrivals follow calibrated profile",
                        service_rule="Added flights retain source service in observed_day mode; otherwise all services are sampled by airline"),
                    service_sampling=service_sampling,
                    single_service_mean=single_mean,
                    single_variability="S_low = mean + 0.5*(S-mean); population mean unchanged, variance quartered",
                    ci_estimand="Mean replication metric; p95_wait CI estimates mean daily p95, not pooled-flight p95",
                    policy_difference="alternative minus baseline", crn=crn)
    write_csv(output / "replications.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_differences.csv", comparisons)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    from .plots import experiment_plots
    experiment_plots(summary, comparisons, cfg, output, synthetic=calibrated or not bool(days))
    from .plots import terminal_policy_plots
    terminal_policy_plots(summary, terminal_scenarios, cfg, output)
    return metadata


def validate_mm1(output, replications=100, seed=553, warmup=5000, duration=20000):
    """Steady-state replication/deletion (Ch. 6), service mean one minute."""
    if replications < 2 or seed < 0 or not np.isfinite(warmup) or warmup < 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid validation parameters")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    for rho in (.5, .7, .85, .95):
        waits = []
        for rep in range(replications):
            arrival_rng, service_rng, _ = streams(seed, rep)
            times = poisson_arrivals(arrival_rng, rho, warmup + duration)
            services = exponential(service_rng, 1, len(times))
            demand = [Flight(str(i), float(a), float(s)) for i, (a, s) in enumerate(zip(times, services))]
            result = simulate(demand, [Gate("G1", "T1")], horizon=warmup + duration)
            observed = [r["wait"] for r in result.flights if r["arrival"] >= warmup]
            if not observed:
                raise ValueError("Observation period too short: no post-warmup arrivals")
            mean = float(np.mean(observed))
            waits.append(mean)
            rows.append(dict(rho=rho, replication=rep, mean_wait=mean, observed_flights=len(observed)))
        ci = confidence_interval(waits)
        theoretical = rho / (1 - rho)
        summary.append(dict(rho=rho, theory=theoretical, **ci,
                            relative_error=(ci["mean"] - theoretical) / theoretical,
                            theory_inside_ci=ci["ci_low"] <= theoretical <= ci["ci_high"]))
        print(f"M/M/1 rho={rho}: simulated={ci['mean']:.4f}, theory={theoretical:.4f}", flush=True)
    write_csv(output / "mm1_replications.csv", rows)
    write_csv(output / "mm1_validation.csv", summary)
    (output / "metadata.json").write_text(json.dumps(dict(seed=seed, replications=replications,
        warmup=warmup, duration=duration, service_mean=1,
        note="Pointwise 95% intervals; occasional noncoverage is expected, not a deterministic test."), indent=2))
    from .plots import mm1_plot
    mm1_plot(summary, output)
    return summary
