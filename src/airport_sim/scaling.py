"""Synthetic capacity/scalability grid with fixed offered demand per terminal."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import platform
import time
import numpy as np
from scipy.stats import gamma
from .inputs import load_days, write_csv
from .metrics import confidence_interval, paired_difference
from .simulator import Flight, Gate, simulate

TERMINALS = [1, 2, 3, 4, 5]
GATES = [1, 5, 10, 20, 50]
POLICIES = [('no_sharing', 'pooled', 0), ('sharing_0', 'sharing', 0), ('sharing_20', 'sharing', 20)]


def calibrate(reference, horizon=1440):
    days = load_days(reference, horizon)
    flights = [f for day in days.values() for f in day]
    terminals = {f.terminal for f in flights}
    services = np.asarray([f.service for f in flights])
    edges = np.linspace(0, horizon, int(np.ceil(horizon / 30)) + 1)
    counts, _ = np.histogram([f.arrival for f in flights], edges)
    mean, variance = float(services.mean()), float(services.var())
    return dict(horizon=horizon, flights_per_terminal_day=len(flights)/len(days)/len(terminals),
                service_mean=mean, service_variance=variance,
                bin_edges=edges.tolist(), arrival_probabilities=(counts/counts.sum()).tolist())


def synthetic_network(profile, terminal_count, seed, replication):
    """Independent identical terminals; terminal prefixes invariant to grid size."""
    flights = []
    for t in range(1, terminal_count+1):
        ar, sr = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, replication, t]).spawn(2)]
        n = int(ar.poisson(profile['flights_per_terminal_day']))
        edges = np.asarray(profile['bin_edges'])
        bins = ar.choice(len(edges)-1, size=n, p=profile['arrival_probabilities'])
        arrivals = edges[bins]+ar.random(n)*(edges[bins+1]-edges[bins])
        mean, variance = profile['service_mean'], profile['service_variance']
        u = np.clip(sr.random(n), np.finfo(float).eps, 1-np.finfo(float).eps)
        services = gamma.ppf(u, mean*mean/variance, scale=variance/mean) if variance else np.full(n, mean)
        flights.extend(Flight(f'T{t}-F{i:05d}', float(a), float(s), f'T{t}', f'A{t}')
                       for i,(a,s) in enumerate(zip(arrivals, services)))
    return flights


def capacity(terminal_count, gates_per_terminal):
    airlines = tuple(f'A{i}' for i in range(1, terminal_count+1))
    shared = (gates_per_terminal+1)//2
    return [Gate(f'T{t}-G{g:03d}', f'T{t}', airlines if g <= shared else (f'A{t}',))
            for t in range(1, terminal_count+1) for g in range(1, gates_per_terminal+1)]


def run_scaling(reference, output, replications=100, seed=553):
    if replications < 2:
        raise ValueError('At least two replications required')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    profile = calibrate(reference)
    rows = []
    for rep in range(replications):
        full = synthetic_network(profile, max(TERMINALS), seed, rep)
        if rep == 0:
            write_csv(output/'first_replication_inputs.csv', [asdict(f) for f in full])
        for nt in TERMINALS:
            flights = [f for f in full if int(f.terminal[1:]) <= nt]
            for ng in GATES:
                gates = capacity(nt, ng)
                for name, policy, transfer in POLICIES:
                    start = time.perf_counter()
                    result = simulate(flights, gates, policy, profile['horizon'], transfer)
                    elapsed = time.perf_counter()-start
                    rows.append(dict(replication=rep, terminals=nt, gates_per_terminal=ng,
                        total_gates=nt*ng, scenario=name, runtime_seconds=elapsed, **result.metrics))
        print(f'Scaling replication {rep+1}/{replications}', flush=True)
    metrics = [k for k in rows[0] if k not in ['replication','terminals','gates_per_terminal','total_gates','scenario']]
    summary, paired = [], []
    for nt in TERMINALS:
        for ng in GATES:
            grouped = {name: [r for r in rows if r['terminals']==nt and r['gates_per_terminal']==ng and r['scenario']==name]
                       for name,_,_ in POLICIES}
            for name,group in grouped.items():
                for metric in metrics:
                    summary.append(dict(terminals=nt,gates_per_terminal=ng,total_gates=nt*ng,scenario=name,metric=metric,
                                        **confidence_interval([r[metric] for r in group])))
            for name in ['sharing_0','sharing_20']:
                for metric in ['mean_wait','p95_wait','utilization','completed_in_day']:
                    paired.append(dict(terminals=nt,gates_per_terminal=ng,baseline='no_sharing',alternative=name,metric=metric,
                        **paired_difference([r[metric] for r in grouped['no_sharing']],[r[metric] for r in grouped[name]])))
    write_csv(output/'replications.csv',rows)
    write_csv(output/'summary.csv',summary)
    write_csv(output/'paired_differences.csv',paired)
    metadata=dict(seed=seed,replications=replications,terminal_counts=TERMINALS,gates_per_terminal=GATES,
        profile=profile,source_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
        input_type='Fully synthetic flights; BTS reference used only for aggregate calibration',
        arrival_model='Poisson daily count per terminal, pooled empirical time-bin mixture with uniform within-bin arrivals',
        terminal_model='Identically distributed independent terminals; no airport-specific mapping',
        shared_gates='ceil(gates_per_terminal / 2); 1, 3, 5, 10, 25',
        common_inputs='Identical flights across capacities and policies; same terminal prefixes across network sizes',
        runtime='Wall-clock simulate() only; input generation and plotting excluded; fixed scenario order, descriptive local timings',
        platform=platform.platform(),machine=platform.machine(),python=platform.python_version())
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    scaling_plots(summary,output)
    return metadata


def scaling_plots(summary,output):
    from .plots import plt, save
    from matplotlib.colors import LogNorm
    lookup={(int(r['terminals']),int(r['gates_per_terminal']),r['scenario'],r['metric']):float(r['mean']) for r in summary}
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(metric,title) in zip(axes,[('mean_wait','Mean wait (minutes)'),('utilization','In-window utilization'),('runtime_seconds','Simulation runtime (ms)')]):
        values=np.array([[lookup[t,g,'no_sharing',metric] for g in GATES] for t in TERMINALS])
        if metric=='runtime_seconds': values*=1000
        shown=values+1 if metric=='mean_wait' else values
        im=ax.imshow(shown,aspect='auto',cmap='YlGnBu',norm=LogNorm(vmin=1,vmax=max(2,shown.max())) if metric=='mean_wait' else None)
        for i in range(5):
            for j in range(5):
                v=values[i,j]
                ax.text(j,i,f'{v:.0%}' if metric=='utilization' else f'{v:.1f}',ha='center',va='center',fontsize=8,
                        color='white' if im.norm(shown[i,j])>.6 else 'black')
        ax.set(xticks=range(5),xticklabels=GATES,yticks=range(5),yticklabels=TERMINALS,
               xlabel='Gates per terminal',ylabel='Number of terminals',title=title)
    fig.suptitle('Synthetic scaling: no sharing; fixed expected demand per terminal (100 replications)')
    save(fig,output,'scaling_grid')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,ng in zip(axes,[1,5,10]):
        for name,_,_ in POLICIES:
            data=[next(r for r in summary if int(r['terminals'])==nt and int(r['gates_per_terminal'])==ng and r['scenario']==name and r['metric']=='mean_wait') for nt in TERMINALS]
            ax.errorbar(TERMINALS,[float(r['mean']) for r in data],yerr=[float(r['ci_high'])-float(r['mean']) for r in data],marker='o',capsize=3,label=name.replace('_',' '))
        ax.set(xlabel='Number of terminals',ylabel='Mean wait (minutes)',title=f'{ng} gate(s) per terminal',xticks=TERMINALS)
        ax.legend(fontsize=8)
    fig.suptitle('Synthetic sharing sensitivity; pointwise 95% confidence intervals')
    save(fig,output,'scaling_sharing')
