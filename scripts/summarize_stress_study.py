"""Audit and plot demand sensitivity after run_stress_study.py finishes."""
from pathlib import Path
import csv
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from airport_sim.plots import plt, save, experiment_plots
from airport_sim.inputs import write_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/stress'
LEVELS = [1,4,5,6]


def rows(path):
    return list(csv.DictReader(path.open()))


def numeric(path):
    data=rows(path)
    for r in data:
        for key in ['n','mean','sd','ci_low','ci_high']:
            r[key]=float(r[key])
    return data


def main():
    combined=[]
    runs={}
    for mode in ['synthetic','bts']:
        for multiplier in LEVELS:
            p=OUT/f'{mode}_20_x{multiplier}'
            metadata=json.loads((p/'metadata.json').read_text())
            assert len(metadata['gates'])==60 and metadata['replications']==100
            assert metadata['config']['demand_multiplier']==multiplier
            runs[mode,multiplier]=rows(p/'replications.csv')
            summary=numeric(p/'summary.csv')
            combined.extend(dict(mode=mode,demand_multiplier=multiplier,**r) for r in summary)
            experiment_plots(summary,numeric(p/'paired_differences.csv'),metadata['config'],p,mode=='synthetic')
    for mode in ['synthetic','bts']:
        for multiplier in LEVELS:
            for baseline,current in zip(runs[mode,1],runs[mode,multiplier]):
                assert baseline['scenario']==current['scenario']
                assert baseline['sampled_date']==current['sampled_date']
                if baseline['scenario'].startswith('single_'):
                    assert baseline==current
                else:
                    assert float(current['n_flights'])==multiplier*float(baseline['n_flights'])
    for multiplier in LEVELS:
        for a,b in zip(runs['synthetic',multiplier],runs['bts',multiplier]):
            assert (a['scenario'],a['sampled_date'],a['n_flights'])==(b['scenario'],b['sampled_date'],b['n_flights'])
    observed=rows(OUT/'bts_20_x5_observed/replications.csv')
    assert len(observed)==len(runs['bts',5])
    assert all(sum(r['scenario']==name for r in observed)==100 for name in {r['scenario'] for r in observed})
    for a,b in zip(runs['bts',5],observed):
        assert (a['scenario'],a['sampled_date'],a['n_flights'])==(b['scenario'],b['sampled_date'],b['n_flights'])
        if a['scenario'].startswith('single_'):
            assert a==b
    write_csv(OUT/'demand_summary.csv',combined)
    lookup={(r['mode'],r['demand_multiplier'],r['scenario'],r['metric']):r for r in combined}
    for mode in ['synthetic','bts']:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,scenarios in zip(axes,[['terminal_dedicated','terminal_pooled','terminal_smoothed'],
                                     ['network_no_sharing','network_sharing_0','network_sharing_40']]):
            for scenario in scenarios:
                values=[lookup[mode,m,scenario,'mean_wait'] for m in LEVELS]
                ax.errorbar(LEVELS,[r['mean'] for r in values],
                    yerr=[r['ci_high']-r['mean'] for r in values],capsize=3,marker='o',label=scenario.replace('_',' '))
            ax.set(xlabel='Demand multiplier (1x = retained reference sample)',ylabel='Mean wait (minutes)',xticks=LEVELS)
            ax.legend(fontsize=8)
        axes[0].set_title('T4: 20 gates');axes[1].set_title('Network: 60 gates')
        fig.suptitle(('Calibrated synthetic' if mode=='synthetic' else 'BTS-based constructed stress tests')+'; 100 replications, 95% CIs')
        save(fig,OUT,f'{mode}_demand_sensitivity')
    (OUT/'audit.json').write_text(json.dumps(dict(replications=100,levels=LEVELS,
        gates_per_terminal=20,terminals=3,flight_counts_aligned=True,
        source_dates_aligned=True,single_gate_unchanged_across_multipliers=True),indent=2)+'\n')
    print('Demand/count/capacity alignment and unchanged single-gate controls verified.')


if __name__=='__main__':main()
