"""Build the aligned original-demand paper and its final synthetic scaling section."""
from pathlib import Path
import csv
import json
import shutil
import hashlib
from datetime import datetime
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT.parent/'report'
MAIN=ROOT/'results/main'
SCALING=ROOT/'results/scaling'


def read(path): return list(csv.DictReader(path.open()))
def summary(mode): return read(MAIN/mode/'summary.csv')
def stat(mode,key): return next(r for r in summary(mode) if r['scenario']==key and r['metric']=='mean_wait')
def val(mode,key): return float(stat(mode,key)['mean'])
def interval(r): return f"[{float(r['ci_low']):.2f}, {float(r['ci_high']):.2f}]"
def table(headers,data,caption):
    return ('\n\\begin{center}\n\\begin{minipage}{\\linewidth}\n\\centering\n\\captionof{table}{'+caption+'}\n\\small\n'
        '\\begin{tabular}{@{}'+'l'+'r'*(len(headers)-1)+'@{}}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'+
        '\n'.join(' & '.join(row)+r' \\' for row in data)+'\n\\bottomrule\n\\end{tabular}\n\\end{minipage}\n\\end{center}\n')
def figure(path,caption,width='1'):
    return ('\n\\begin{center}\n\\begin{minipage}{\\linewidth}\n\\centering\n\\includegraphics[width='+width+r'\linewidth]{'+path+
        '}\n\\captionof{figure}{'+caption+'}\n\\end{minipage}\n\\end{center}\n')
def replace_section(s,start,end,body):
    a=s.index(start);b=s.index(end,a)
    return s[:a]+start+'\n'+body+'\n'+s[b:]


def results(mode):
    names=[('Dedicated terminal','terminal_dedicated'),('Pooled terminal','terminal_pooled'),('Smoothed terminal','terminal_smoothed'),
        ('Network, no sharing','network_no_sharing'),('Sharing, 0 min','network_sharing_0'),('Sharing, 10 min','network_sharing_10'),
        ('Sharing, 20 min','network_sharing_20'),('Sharing, 40 min','network_sharing_40'),('One gate closed','network_closed_gate'),('Initially occupied','network_initial_occupied')]
    text=table(['Policy','Mean wait (min)',r'95\% CI'],[[label,f'{val(mode,key):.2f}',interval(stat(mode,key))] for label,key in names],
        'Original demand, four gates per terminal, 100 replications. No demand multiplication is applied.')
    text+=figure(f'results_main/{mode}/experiment_overview.pdf',r'Aligned single-gate, T4 single-terminal, and three-terminal network results; replication means with 95\% confidence intervals.')
    pairs=read(MAIN/mode/'paired_differences.csv')
    data=[]
    for rho in [.5,.7,.85,.95]:
        a=f'single_{rho:g}_base';b=f'single_{rho:g}_low_variability'
        d=next(r for r in pairs if r['alternative']==b and r['metric']=='mean_wait')
        data.append([f'{rho:.2f}',f'{val(mode,a):.2f}',f'{val(mode,b):.2f}',f"{float(d['mean']):.2f}",interval(d)])
    text+=table([r'$\rho$','Original','Lower variability','Difference',r'95\% paired CI'],data,'Single-gate mean waiting in minutes. This setting always uses one gate.')
    text+=figure(f'results_main/{mode}/policy_differences.pdf','Paired policy differences: alternative minus baseline, mean waiting in minutes.','.74')
    return text


def discussion(mode):
    d=next(r for r in read(MAIN/mode/'paired_differences.csv') if r['alternative']=='network_sharing_0' and r['metric']=='mean_wait')
    return (f'With four gates per terminal and the original daily counts, network waiting is {val(mode,"network_no_sharing"):.2f} minutes without sharing, '
        f'{val(mode,"network_sharing_0"):.2f} minutes with zero-transfer sharing, and {val(mode,"network_sharing_40"):.2f} minutes with a 40-minute transfer. '
        f'The paired zero-transfer effect is {float(d["mean"]):.2f} minutes (95\\% CI {interval(d)}). '
        f'T4 pooling gives {val(mode,"terminal_pooled"):.2f} minutes; smoothing gives {val(mode,"terminal_smoothed"):.2f} minutes. '
        'Terminal imbalance and time-of-day peaks cause congestion even when average network offered load is below one. '
        'These finite-day results include queue draining after the observation horizon and do not establish steady-state stability.\n\n'
        'The single-gate variability reduction improves mean waiting at each tested load. Policy comparisons share pre-generated flights within a replication. '
        'The transfer-time grid does not establish a break-even point beyond the tested range.\n')


def scaling_section():
    rows=read(SCALING/'summary.csv');meta=json.loads((SCALING/'metadata.json').read_text())
    lookup={(int(r['terminals']),int(r['gates_per_terminal']),r['scenario'],r['metric']):r for r in rows}
    def value(t,g,scenario='no_sharing',metric='mean_wait'):return float(lookup[t,g,scenario,metric]['mean'])
    profile=meta['profile'];demand=profile['flights_per_terminal_day'];mean=profile['service_mean']
    text=r'''\clearpage
\section*{Final Experiment: Synthetic Scaling Study}
This separate final experiment varies network size and gate capacity using fully synthetic traffic. It does not add flights to the BTS case study. The full factorial grid has 1, 2, 3, 4, and 5 terminals crossed with 1, 5, 10, 20, and 50 gates per terminal: 25 configurations. Each configuration runs 100 replications for no sharing, zero-transfer sharing, and 20-minute-transfer sharing, totaling 7,500 policy replications. The main two-part study above remains fixed at three terminals and four gates per terminal.

\textbf{Controlled synthetic demand.} Terminals are independent and identically distributed. Each terminal draws a Poisson daily flight count with a constant mean, then assigns each arrival to a bin from the pooled BTS half-hour profile and draws uniformly inside that bin. Services follow a Gamma distribution matched to the pooled reference mean and population variance. The original carrier-to-terminal mapping is not used here; labels T1--T5 denote generic synthetic terminals. The source data calibrate only aggregate synthetic parameters.
'''
    text+=f'Expected demand is {demand:.2f} flights per terminal per day, and mean service is {mean:.2f} minutes. Expected total demand therefore grows from {demand:.2f} to {5*demand:.2f} flights per day as terminal count increases from one to five. Demand per terminal does not increase when gates are added.\n'
    text+=r'''
\textbf{Capacity and random-number controls.} Every terminal has the specified gate count; total gates range from 1 to 250. For sharing policies, $\lceil c/2\rceil$ gates per terminal accept every synthetic airline, giving 1, 3, 5, 10, and 25 shareable gates at the five capacity levels; other gates accept only the local airline. This rounding rule makes the one-gate sharing case meaningful. The no-sharing policy remains local regardless of compatibility. Flights are identical across capacities and policies within a replication. Terminal-specific random streams make the first $T$ terminals' flights unchanged when a larger network is constructed. All scenarios use an empty start, a 1,440-minute observation day, no closures, and subsequent draining.
'''
    text+=table(['Gates per terminal','Expected flights/terminal/day','Offered load'],[[str(g),f'{demand:.2f}',f'{100*demand*mean/(1440*g):.1f}\\%'] for g in [1,5,10,20,50]],'Offered load per terminal is constant across terminal counts at a given gate count.')
    text+=table(['Terminals','1 gate','5 gates','10 gates','20 gates','50 gates'],[[str(t)]+[f'{value(t,g):.2f}' for g in [1,5,10,20,50]] for t in range(1,6)],'All 25 configurations: no-sharing mean wait in minutes, averaged over 100 replications.')
    text+=figure('results_scaling/scaling_grid.pdf','Synthetic scaling grid. Cell labels show arithmetic mean metrics; waiting colors use a logarithmic scale of one plus waiting. Runtime measures only the simulator call, in milliseconds.')
    text+=figure('results_scaling/scaling_sharing.pdf',r'Sharing sensitivity at 1, 5, and 10 gates per terminal, with pointwise 95\% confidence intervals. Complete results for all 25 configurations are stored in CSV.')
    text+=f'At five terminals, no-sharing mean waiting decreases from {value(5,1):.2f} minutes with one gate per terminal to {value(5,5):.2f} minutes with five gates and {value(5,10):.2f} minutes with ten gates. The one-gate case has offered load above one and long post-day drainage; its waiting values are not steady-state estimates. The 20- and 50-gate configurations show zero no-sharing waiting in the sampled runs. This is not proof of zero delay for every future random draw.\n\n'
    text+=f'With five gates per terminal and five terminals, sharing changes mean waiting from {value(5,5):.2f} minutes to {value(5,5,"sharing_0"):.2f} minutes at zero transfer and {value(5,5,"sharing_20"):.2f} minutes at a 20-minute transfer. '
    text+=f'At one gate per terminal and five terminals, a 20-minute transfer instead raises mean waiting from {value(5,1):.2f} to {value(5,1,"sharing_20"):.2f} minutes. Greedy sharing reserves already-scarce capacity during travel, so sharing is not uniformly beneficial. '
    text+=r'''With one terminal, sharing is exactly equivalent to the local policy because there is no other terminal. Under no sharing, terminal count should not systematically change the population waiting distribution when per-terminal demand is fixed; small differences across rows reflect finite simulation samples. Sharing introduces interactions and can change the effect of network size.

\textbf{Timing interpretation and reproducibility.} Elapsed wall time covers only each \texttt{simulate()} call, excluding input generation and plotting. These are descriptive timings on the local machine and may be affected by concurrent processes; they are not a controlled hardware benchmark or an asymptotic complexity claim. The full replication metrics, Student-$t$ summaries, paired differences, machine information, calibration parameters, and first-replication input flights are stored under \texttt{results\_scaling}. Run \texttt{scripts/run\_scaling.py} to reproduce the grid; \texttt{scripts/build\_current\_report.py} restores the aligned main results and appends this final experiment.
'''
    return text



def update_service_definition(s, reference):
    durations=np.asarray([float(r['service']) for r in reference])
    mean=float(durations.mean())
    days=len({r['date'] for r in reference})
    loads={t:sum(float(r['service']) for r in reference if r['terminal']==t)/(days*4*1440)
           for t in ['T4','T5','T8']}
    s=s.replace('105.6724',f'{mean:.4f}').replace('105.67',f'{mean:.2f}').replace('52.84',f'{mean/2:.2f}')
    for old,t in [('86.29','T4'),('101.88','T5'),('41.27','T8')]:
        s=s.replace(old+r'\%',f'{100*loads[t]:.2f}'+r'\%')
    s=s.replace(r'76.48\%',f'{100*sum(loads.values())/3:.2f}'+r'\%')
    s=s.replace('Service includes the clearance buffer and ends',
                'Service is the matched departure-minus-arrival interval, with no added buffer, and ends')
    s=s.replace('service time is that stay plus a 5-minute clearance buffer.',
        r'service time is exactly that stay: $S_i=t_{i,\mathrm{next\ gate\ departure}}-t_{i,\mathrm{gate\ arrival}}$. No clearance buffer is added. Both timestamps refer to the same aircraft at JFK and are retained in UTC in the cleaned CSV. The service interval uses gate arrival time; the modeled request time is defined separately above.')
    s=s.replace('Service is screened turnaround plus the 5-minute buffer.',
                'Service is the screened departure-minus-arrival interval, with no added buffer.')
    lines=s.splitlines()
    for i,line in enumerate(lines):
        for airline,terminal in [('AA','T8'),('B6','T5'),('DL','T4')]:
            if line.startswith(f'{airline} & {terminal} &'):
                values=np.asarray([float(r['service']) for r in reference if r['airline']==airline])
                lines[i]=f'{airline} & {terminal} & {len(values):,} & {values.mean():.2f} & {values.std(ddof=1)/values.mean():.3f} & {loads[terminal]:.3f} '+r'\\'
    s='\n'.join(lines)+'\n'
    fits=json.loads((ROOT/'results/inputs_current/input_fit.json').read_text())
    best=min(fits['fits'],key=lambda f:f['aic'])['distribution']
    labels={'lognorm':'Lognormal','gamma':'Gamma','expon':'Exponential'}
    s=s.replace('Lognormal has the lowest training AIC',labels[best]+' has the lowest training AIC')
    s=s.replace('the lognormal Q--Q plot','the '+labels[best].lower()+' Q--Q plot')
    lines=s.splitlines()
    for i,line in enumerate(lines):
        for f in fits['fits']:
            if line.startswith(f['distribution']+' &'):
                lines[i]=f"{f['distribution']} & {f['aic']:.2f} & {f['ks_statistic']:.3f} & {f['holdout_ks_statistic']:.3f} "+r'\\'
    s='\n'.join(lines)+'\n'
    s=s.replace('figures_bts/input_diagnostics.pdf','results_inputs/input_diagnostics.pdf')
    s=s.replace(r'Original cleaned data and provenance remain in \texttt{results\_aligned}.',
                r'Current cleaned data, UTC arrival/departure timestamps, input diagnostics, and provenance are in \texttt{results\_inputs}.')
    return s


def main():
    # Audit original demand and alignment before changing report outputs.
    a=read(MAIN/'synthetic/replications.csv');b=read(MAIN/'bts/replications.csv')
    assert len(a)==len(b)==2000
    reference=read(ROOT/'data/processed/jfk_2025_07.csv')
    source_hash=hashlib.sha256((ROOT/'data/processed/jfk_2025_07.csv').read_bytes()).hexdigest()
    for row in reference:
        elapsed=(datetime.fromisoformat(row['next_gate_departure_utc'])-datetime.fromisoformat(row['actual_gate_arrival_utc'])).total_seconds()/60
        assert float(row['service'])==float(row['turnaround_proxy'])==elapsed
    assert json.loads((SCALING/'metadata.json').read_text())['source_sha256']==source_hash
    counts={d:sum(r['date']==d for r in reference) for d in {r['date'] for r in reference}}
    for x,y in zip(a,b):
        assert (x['scenario'],x['sampled_date'],x['n_flights'])==(y['scenario'],y['sampled_date'],y['n_flights'])
        if x['scenario']=='network_no_sharing':assert float(x['n_flights'])==counts[x['sampled_date']]
    for mode in ['synthetic','bts','bts_observed']:
        m=json.loads((MAIN/mode/'metadata.json').read_text())
        assert m['input_sha256']==source_hash
        assert m['config']['gates_per_terminal']==4 and m['config']['demand_multiplier']==1
        assert m['replications']==100 and len(m['gates'])==12
        actual=read(MAIN/mode/'replications.csv')
        for scenario in {r['scenario'] for r in actual}:
            reps=[int(r['replication']) for r in actual if r['scenario']==scenario]
            assert len(reps)==100 and set(reps)==set(range(100))
        assert all(int(r['n'])==100 for r in summary(mode))
    grid=read(SCALING/'replications.csv');assert len(grid)==7500
    for rep in range(100):
        subset=[r for r in grid if int(r['replication'])==rep]
        for t in range(1,6):
            group=[r for r in subset if int(r['terminals'])==t]
            assert len(group)==15 and len({r['n_flights'] for r in group})==1
    (SCALING/'audit.json').write_text(json.dumps(dict(configurations=25,policy_replications=7500,
        flight_counts_constant_across_capacities_and_policies=True,main_demand_multiplier=1,main_gates_per_terminal=4),indent=2)+'\n')
    validation=ROOT/'results/validation_100'
    validation_meta=json.loads((validation/'metadata.json').read_text())
    validation_rows=read(validation/'mm1_validation.csv')
    assert validation_meta['replications']==100 and all(int(r['n'])==100 for r in validation_rows)
    shutil.copytree(validation,REPORT/'results_validation',dirs_exist_ok=True)
    shutil.copytree(MAIN,REPORT/'results_main',dirs_exist_ok=True)
    shutil.copytree(SCALING,REPORT/'results_scaling',dirs_exist_ok=True)
    inputs=REPORT/'results_inputs'
    shutil.copytree(ROOT/'results/inputs_current',inputs,dirs_exist_ok=True)
    for name in ['jfk_2025_07.csv','bts_provenance.json']:
        shutil.copy2(ROOT/'data/processed'/name,inputs/name)
    s=(REPORT/'archive_20gate_reference/report.tex').read_text()
    s=s.replace('30 replications', '100 replications')
    s=s.replace('Stationary $M/M/1$ verification, unchanged by the case-study capacity update.', 'Stationary $M/M/1$ verification: 100 independent replications at each utilization.')
    s=s.replace('figures/mm1_validation.pdf', 'results_validation/mm1_validation.pdf')
    inside=sum(float(r['ci_low'])<=float(r['theory'])<=float(r['ci_high']) for r in validation_rows)
    s=s.replace('All four theoretical mean waits lie within their pointwise simulation intervals.',
        f'{inside} of the four theoretical mean waits lie within their pointwise 95\\% simulation intervals; pointwise interval coverage is not a deterministic correctness test.')
    s=s.replace('20 assumed gates per terminal','four assumed gates per terminal').replace('20 gates per terminal','four gates per terminal')
    s=s.replace('twenty gates','four gates').replace('Twenty gates','Four gates').replace('sixty for the network','twelve for the network')
    s=s.replace('sixty','twelve').replace('60 total gates','12 total gates').replace('all 60 gates','all 12 gates')
    s=s.replace('Four gates means twenty at each terminal, not twenty in the entire network.', 'Each terminal has four gates, giving twelve gates across the three-terminal network.')
    s=s.replace('Ten gates at each terminal','Two gates at each terminal').replace('ten shared-compatible gates','two shared-compatible gates')
    s=s.replace('the remaining ten','the remaining two').replace('ten of which','two of which')
    s=s.replace('T4 & 20 &','T4 & 4 &').replace('T4, T5, T8 & 60 &','T4, T5, T8 & 12 &')
    s=s.replace('$20\\times1440$','$4\\times1440$').replace('17.26\\%','86.29\\%').replace('20.38\\%','101.88\\%').replace('8.25\\%','41.27\\%').replace('15.30\\%','76.48\\%')
    s=s.replace('Offered load, 20 gates','Offered load, 4 gates').replace('& 0.083','& 0.413').replace('& 0.204','& 1.019').replace('& 0.173','& 0.863')
    s=s.replace('20-gate scenarios','four-gate scenarios').replace('Seventeen automated tests pass','Twenty automated tests pass')
    abstract=(f'We compare calibrated synthetic inputs and BTS-derived inputs under aligned original-demand settings: three terminals with four assumed gates each, two per terminal supporting cross-terminal sharing. '
        'The reference sample contains 3,877 screened JFK domestic turnarounds over 31 days in July 2025. Synthetic inputs preserve the sampled daily counts by airline and match empirical service means and variances; the BTS baseline uses observed request times and by-airline empirical service resampling. No demand multiplication is used. '
        f'Over 100 replications, zero-transfer sharing changes mean network waiting from {val("synthetic","network_no_sharing"):.2f} to {val("synthetic","network_sharing_0"):.2f} minutes with synthetic inputs and from {val("bts","network_no_sharing"):.2f} to {val("bts","network_sharing_0"):.2f} minutes with BTS-derived inputs. '
        'A separate final synthetic experiment crosses 1--5 terminals with 1, 5, 10, 20, and 50 gates per terminal, holding expected demand per terminal fixed. Capacities and gate occupancy remain modeling assumptions; simulated waiting is not measured JFK delay.\n\n'+r'\textbf{Keywords:} discrete-event simulation; gate queues; BTS; common random numbers; synthetic scaling.'+'\n')
    s=replace_section(s,r'\section*{Abstract}',r'\section*{Common experimental settings}',abstract)
    s=replace_section(s,r'\section*{I.5\quad Results and interpretation}',r'\section*{I.6\quad Discussion}',results('synthetic'))
    s=replace_section(s,r'\section*{I.6\quad Discussion}',r'\clearpage',discussion('synthetic'))
    s=replace_section(s,r'\section*{II.4\quad Results}',r'\section*{II.5\quad Interpretation and limitations}',results('bts'))
    observed=table(['Policy','Resampling','Observed-day replay',r'Replay 95\% CI'],[[label,f'{val("bts",key):.2f}',f'{val("bts_observed",key):.2f}',interval(stat('bts_observed',key))]
        for label,key in [('T4 pooled','terminal_pooled'),('No sharing','network_no_sharing'),('Sharing, 0 min','network_sharing_0'),('Sharing, 40 min','network_sharing_40')]],
        'Sensitivity to service dependence at the original flight counts and four gates per terminal. Replay preserves each sampled day\'s original services and requests.')
    limitations=r'''
The original-data study uses only retained observed flights, with no added copies. The empirical-resampling baseline still generates service assignments from real service samples; observed-day replay retains the original matched service values. Both use turnaround proxies rather than measured gate occupancy. There are no observed gate identifiers or complete gate trajectories. Tail-matched stays may include towing, and selected domestic carriers and same-day screening omit traffic. Capacities and compatibility are assumptions. Thus this case study does not establish actual JFK delays or an optimal airport gate count. Confidence intervals quantify simulation variation conditional on the selected inputs, excluding modeling and screening uncertainty.
'''
    s=replace_section(s,r'\section*{II.5\quad Interpretation and limitations}',r'\section*{II.6\quad Conclusion and reproducibility}',discussion('bts')+observed+limitations)
    conclusion=r'''
The main synthetic and BTS studies use the same original demand counts, service moments, terminal mapping, and four-gate capacity. Sharing and pooling are evaluated with paired inputs under these aligned assumptions. The following final experiment studies capacity and network size with separate, fully synthetic traffic; it does not expand the BTS records.

The \texttt{main\_synthetic\_4.json}, \texttt{main\_bts\_4.json}, and \texttt{main\_bts\_observed\_4.json} configurations reproduce the main studies with 100 replications and seed 553. The report's \texttt{results\_main} directory stores results, metadata, figures, and first-replication traces. Original cleaned data and provenance remain in \texttt{results\_aligned}. Superseded demand-expansion results are archived separately and are not part of the current main analysis.
'''
    s=replace_section(s,r'\section*{II.6\quad Conclusion and reproducibility}',r'\begin{thebibliography}{9}',conclusion)
    s=update_service_definition(s,reference)
    replication_table=table(['Experiment','Independent replications'],[
        ['Calibrated synthetic main study','100 per policy'],['BTS main study','100 per policy'],
        ['Observed-day sensitivity','100 per policy'],['Synthetic scaling grid','100 per configuration/policy'],
        [r'$M/M/1$ verification',r'100 per $\rho$']],
        'All current experiments use 100 independent replications per condition.')
    s=s.replace(r'\part*{Part I: Calibrated Synthetic Experiments}',replication_table+r'\part*{Part I: Calibrated Synthetic Experiments}')
    first=s.index(r'\part*{Part I:');second=s.index(r'\part*{Part II:');bib=s.index(r'\begin{thebibliography}')
    common=s[:first];refs=s[bib:]
    (REPORT/'report_synthetic.tex').write_text(common+s[first:second].replace('\\clearpage\n','')+refs)
    (REPORT/'report_bts.tex').write_text(common+s[second:bib]+refs)
    (REPORT/'report.tex').write_text(s[:bib]+scaling_section()+refs)
    print('Original 4-gate inputs aligned; 25 synthetic scaling configurations audited; reports updated.')


if __name__=='__main__':main()
