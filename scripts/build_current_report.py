"""Build the aligned original-demand paper and its final synthetic scaling section."""
from pathlib import Path
import csv
import json
import shutil
import hashlib
from datetime import datetime
import numpy as np
from scipy.stats import t as student_t

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


def crn_section(mode):
    records=read(MAIN/mode/'replications.csv')
    def values(scenario):
        chosen=sorted((r for r in records if r['scenario']==scenario),key=lambda r:int(r['replication']))
        assert [int(r['replication']) for r in chosen]==list(range(100))
        return np.asarray([float(r['mean_wait']) for r in chosen])
    dedicated=values('terminal_dedicated')
    paired=values('terminal_pooled')-dedicated
    independent=values('terminal_pooled_independent')-dedicated
    raw=[]
    for label,data in [('Common random numbers',paired),('Independent inputs',independent)]:
        variance=float(data.var(ddof=1));mean=float(data.mean());n=len(data)
        half=float(student_t.ppf(.975,n-1)*np.sqrt(variance/n))
        raw.append(dict(method=label,n=n,mean_difference=mean,replication_difference_variance=variance,
                        mean_estimator_variance=variance/n,ci_low=mean-half,ci_high=mean+half,ci_half_width=half))
    meta=json.loads((MAIN/mode/'metadata.json').read_text())['crn']
    assert np.isclose(raw[0]['replication_difference_variance'],meta['paired_difference_variance'])
    assert np.isclose(raw[1]['replication_difference_variance'],meta['independent_difference_variance'])
    ratio=raw[1]['replication_difference_variance']/raw[0]['replication_difference_variance']
    assert np.isclose(ratio,meta['variance_ratio_independent_over_crn'])
    with (REPORT/'results_main'/mode/'crn_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    text=r'''
\textbf{Statistical variance reduction: common random numbers (CRN).}
For T4 pooled-minus-dedicated mean waiting, we compare two sampling designs with 100 replications each. The CRN design gives both policies the same sampled date, arrivals, and flight-specific services: $D_r=P(U_r)-D(U_r)$. The independent-input comparator uses a separately seeded day and service realization for the pooled alternative, while retaining the same dedicated baseline: $D_r^{\mathrm{ind}}=P(U'_r)-D(U_r)$. Both target the same expected policy difference; they need not have identical finite-sample means.
\[
\operatorname{Var}(P-D)=\operatorname{Var}(P)+\operatorname{Var}(D)-2\operatorname{Cov}(P,D),
\qquad \widehat{\operatorname{Var}}(\overline D)=s_D^2/100.
\]
'''
    text+=table(['Sampling design',r'Mean $D$ (min)',r'95\% CI',r'$s_D^2$ (min$^2$)',r'$s_D^2/100$'],
        [[r['method'],f"{r['mean_difference']:.2f}",interval(r),f"{r['replication_difference_variance']:.2f}",f"{r['mean_estimator_variance']:.3f}"] for r in raw],
        'CRN precision comparison for T4 pooling versus dedicated queues. Variances describe the policy-difference statistic, not individual flight service times.')
    text+=(f'The observed independent-to-CRN variance ratio is {ratio:.2f}; the estimated variance of the mean policy difference is '
           f'{100*(1-1/ratio):.2f}\\% lower with CRN. At the same 100 replications, the independent-input confidence-interval half-width is '
           f'{np.sqrt(ratio):.2f} times the CRN half-width. This is a comparison of sample variances conditional on this input model, '
           'not a guaranteed efficiency factor or a measured reduction in airport delays.\n\n')
    comparisons=[]
    metadata=json.loads((MAIN/mode/'metadata.json').read_text())
    for terminal,names in metadata['terminal_scenarios'].items():
        baseline=values(names['dedicated'])
        pv=float((values(names['pooled'])-baseline).var(ddof=1))
        iv=float((values(names['pooled_independent'])-baseline).var(ddof=1))
        assert np.isclose(pv,metadata['crn_by_terminal'][terminal]['paired_difference_variance'])
        assert np.isclose(iv,metadata['crn_by_terminal'][terminal]['independent_difference_variance'])
        comparisons.append(dict(terminal=terminal,n=100,crn_variance=pv,independent_variance=iv,ratio=iv/pv if pv else None))
    with (REPORT/'results_main'/mode/'crn_by_terminal_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    text+=table(['Terminal',r'CRN $s_D^2$',r'Independent $s_D^2$','Variance ratio'],[
        [r['terminal'],f"{r['crn_variance']:.2f}",f"{r['independent_variance']:.2f}",f"{r['ratio']:.2f}" if r['ratio'] is not None else 'Undefined'] for r in comparisons],
        'CRN comparison for each terminal: pooled-minus-dedicated mean waiting, 100 replications. Ratios are independent variance divided by CRN variance.')
    text+=r'''\textbf{Distinction from lower service variability.}
The single-gate intervention $S'=m+0.5(S-m)$ changes the modeled service distribution, preserving its population mean and reducing its variance to one quarter. It can change actual simulated waiting. CRN instead changes how two policy runs are coupled; it preserves each policy's marginal input model and aims to estimate their difference more precisely. These are separate experiments, and both are retained in this report.
'''
    return text


def all_terminal_section(mode):
    meta=json.loads((MAIN/mode/'metadata.json').read_text())
    pairs=read(MAIN/mode/'paired_differences.csv')
    text=r'\textbf{Separate single-terminal results for T4, T5, and T8.}'+'\n'
    text+='Each terminal is simulated on its own with four local gates and its own flights. These local experiments do not borrow gates from another terminal. All three policies share the same local input flights within a replication; each condition has 100 replications.\n'
    data=[]
    for terminal,names in meta['terminal_scenarios'].items():
        for policy in ['dedicated','pooled','smoothed']:
            r=stat(mode,names[policy])
            data.append([terminal,policy.capitalize(),f"{float(r['mean']):.2f}",interval(r)])
    text+=table(['Terminal','Local policy','Mean wait (min)',r'95\% CI'],data,
                'Each terminal separately: four gates and 100 replications per local policy.')
    text+=figure(f'results_main/{mode}/all_terminal_policies.pdf',
                r'Single-terminal policy comparison across T4, T5, and T8. Error bars are pointwise 95\% confidence intervals.')
    data=[]
    for terminal,names in meta['terminal_scenarios'].items():
        for baseline,alternative,label in [('dedicated','pooled','Pooling - dedicated'),('pooled','smoothed','Smoothing - pooling')]:
            r=next(r for r in pairs if r['baseline']==names[baseline] and r['alternative']==names[alternative] and r['metric']=='mean_wait')
            data.append([terminal,label,f"{float(r['mean']):.2f}",interval(r)])
    text+=table(['Terminal','Comparison','Mean difference (min)',r'95\% paired CI'],data,
                'Within-terminal policy effects with common random numbers. Differences are alternative minus baseline.')
    return text


def results(mode):
    names=[('T4 dedicated','terminal_dedicated'),('T4 pooled','terminal_pooled'),('T4 smoothed','terminal_smoothed'),
        ('Network, no sharing','network_no_sharing'),('Sharing, 0 min','network_sharing_0'),('Sharing, 10 min','network_sharing_10'),
        ('Sharing, 20 min','network_sharing_20'),('Sharing, 40 min','network_sharing_40'),('One gate closed','network_closed_gate'),('Initially occupied','network_initial_occupied')]
    text=table(['Policy','Mean wait (min)',r'95\% CI'],[[label,f'{val(mode,key):.2f}',interval(stat(mode,key))] for label,key in names],
        'Original demand, four gates per terminal, 100 replications. No demand multiplication is applied.')
    text+=figure(f'results_main/{mode}/experiment_overview.pdf',r'Single-gate, reference T4, and network overview. Separate results for all three terminals follow; error bars show 95\% confidence intervals.')
    text+=all_terminal_section(mode)
    pairs=read(MAIN/mode/'paired_differences.csv')
    data=[]
    for rho in [.5,.7,.85,.95]:
        a=f'single_{rho:g}_base';b=f'single_{rho:g}_low_variability'
        d=next(r for r in pairs if r['alternative']==b and r['metric']=='mean_wait')
        data.append([f'{rho:.2f}',f'{val(mode,a):.2f}',f'{val(mode,b):.2f}',f"{float(d['mean']):.2f}",interval(d)])
    text+=table([r'$\rho$','Original','Lower variability','Difference',r'95\% paired CI'],data,'Single-gate mean waiting in minutes. This setting always uses one gate.')
    text+=figure(f'results_main/{mode}/policy_differences.pdf','Paired policy differences: alternative minus baseline, mean waiting in minutes.','.66')
    text+=crn_section(mode)
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
    counts={d:sum(r['date']==d for r in reference) for d in {r['date'] for r in reference}}
    terminal_counts=[sum(r['terminal']==t for r in reference) for t in ['T4','T5','T8']]
    sampled=read(MAIN/'bts/replications.csv')
    sampled_mean=np.mean([float(r['n_flights']) for r in sampled if r['scenario']=='network_no_sharing'])
    start=s.index('T4/DL, T5/B6, and T8/AA contain')
    end=s.index('\n',start)
    s=s[:start]+(f'T4/DL, T5/B6, and T8/AA contain {terminal_counts[0]:,}, {terminal_counts[1]:,}, and {terminal_counts[2]:,} reference visits, respectively; '
        f'their average daily counts are {terminal_counts[0]/days:.2f}, {terminal_counts[1]/days:.2f}, and {terminal_counts[2]/days:.2f}. '
        f'Network demand averages {len(reference)/days:.2f} flights per reference day (range {min(counts.values())}--{max(counts.values())}); '
        f'the common 100-day bootstrap realization averages {sampled_mean:.2f}.')+s[end:]
    s=s.replace('3,877',f'{len(reference):,}').replace('between 72 and 147',f'between {min(counts.values())} and {max(counts.values())}')
    s=s.replace('This part uses observed gate-request times from the screened BTS sample and by-airline empirical service resampling.',
        "This part uses observed arrival times and each flight's own matched service duration from the screened BTS sample. The complete selected day is replayed without independently resampling services.")
    s=s.replace('gate-request time is defined as its reconstructed gate-arrival time minus reported taxi-in time plus an assumed 10-minute base taxi time.',
        r'gate-request time is defined as its reconstructed gate-arrival time minus reported taxi-in time: $a_i=t_{i,\mathrm{gate\ arrival}}-\mathrm{TaxiIn}_i$. The previous 10-minute allowance is removed. This gives a touchdown-time proxy; taxi traversal is not a separate modeled stage.')
    s=s.replace("Accordingly, we report a by-airline empirical-resampling baseline and a second mode that replays each selected day's observed service values together with its arrivals.",
        "Accordingly, the main baseline replays each selected day's observed service values together with its arrivals. By-airline empirical service resampling is retained only as a sensitivity comparison.")
    s=s.replace("Each service is sampled independently from the corresponding airline's full July pool. This baseline preserves observed arrival clustering but removes service ordering within the day. An additional observed-day replay preserves each sampled day's original services and arrivals together.",
        "Each flight retains its own matched gate-departure-minus-gate-arrival service duration. Thus the main baseline preserves observed arrival clustering and the within-day association of arrivals and services. Across the 100 replications, dates are sampled with replacement from the 31 observed dates; services are not reassigned. A separate sensitivity samples services independently from the corresponding airline's full July pool.")
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
    ntrain=sum(r['date'] in fits['training_dates'] for r in reference)
    nhold=sum(r['date'] in fits['holdout_dates'] for r in reference)
    s=s.replace('3,024 observations',f'{ntrain:,} observations').replace('853 holdout observations',f'{nhold:,} holdout observations')
    correlations=[r['lag1'] for r in fits['daily_lag1']]
    s=s.replace('$-0.053$ to $0.641$',f'${min(correlations):.3f}$ to ${max(correlations):.3f}$')
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
    assert len(a)==len(b)==2800
    reference=read(ROOT/'data/processed/jfk_2025_07.csv')
    provenance=json.loads((ROOT/'data/processed/bts_provenance.json').read_text())
    assert provenance['cleaning_audit']['baseline_taxi']==0
    assert provenance['cleaning_audit']['clearance']==0
    assert json.loads((MAIN/'bts/metadata.json').read_text())['config']['service_mode']=='observed_day'
    originals={r['id']:r for r in reference}
    for row in read(MAIN/'bts/flight_records/network_no_sharing.csv'):
        original=originals[row['id']]
        assert float(row['arrival'])==float(original['arrival'])
        assert float(row['service'])==float(original['service'])
    source_hash=hashlib.sha256((ROOT/'data/processed/jfk_2025_07.csv').read_bytes()).hexdigest()
    for row in reference:
        elapsed=(datetime.fromisoformat(row['next_gate_departure_utc'])-datetime.fromisoformat(row['actual_gate_arrival_utc'])).total_seconds()/60
        assert float(row['service'])==float(row['turnaround_proxy'])==elapsed
    assert json.loads((SCALING/'metadata.json').read_text())['source_sha256']==source_hash
    counts={d:sum(r['date']==d for r in reference) for d in {r['date'] for r in reference}}
    for x,y in zip(a,b):
        assert (x['scenario'],x['sampled_date'],x['n_flights'])==(y['scenario'],y['sampled_date'],y['n_flights'])
        if x['scenario']=='network_no_sharing':assert float(x['n_flights'])==counts[x['sampled_date']]
    for mode in ['synthetic','bts','bts_resampled']:
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
    for mode in ['synthetic','bts','bts_resampled']:
        shutil.copytree(MAIN/mode,REPORT/'results_main'/mode,dirs_exist_ok=True)
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
        f'The reference sample contains {len(reference):,} screened JFK domestic turnarounds over 31 days in July 2025. Synthetic inputs preserve the sampled daily counts by airline and match empirical service means and variances; the BTS baseline retains each selected flight\'s observed arrival and matched service duration. Arrival is gate arrival minus TaxiIn, with no added taxi allowance. No demand multiplication is used. '
        f'Over 100 replications, zero-transfer sharing changes mean network waiting from {val("synthetic","network_no_sharing"):.2f} to {val("synthetic","network_sharing_0"):.2f} minutes with synthetic inputs and from {val("bts","network_no_sharing"):.2f} to {val("bts","network_sharing_0"):.2f} minutes with BTS-derived inputs. '
        'A separate final synthetic experiment crosses 1--5 terminals with 1, 5, 10, 20, and 50 gates per terminal, holding expected demand per terminal fixed. Capacities and gate occupancy remain modeling assumptions; simulated waiting is not measured JFK delay.\n\n'+r'\textbf{Keywords:} discrete-event simulation; gate queues; BTS; common random numbers; synthetic scaling.'+'\n')
    s=replace_section(s,r'\section*{Abstract}',r'\section*{Common experimental settings}',abstract)
    s=replace_section(s,r'\section*{I.5\quad Results and interpretation}',r'\section*{I.6\quad Discussion}',results('synthetic'))
    s=replace_section(s,r'\section*{I.6\quad Discussion}',r'\clearpage',discussion('synthetic'))
    s=replace_section(s,r'\section*{II.4\quad Results}',r'\section*{II.5\quad Interpretation and limitations}',results('bts'))
    observed=table(['Policy','Observed baseline','Resampling',r'Resampling 95\% CI'],[[label,f'{val("bts",key):.2f}',f'{val("bts_resampled",key):.2f}',interval(stat('bts_resampled',key))]
        for label,key in [('T4 pooled','terminal_pooled'),('T5 pooled','terminal_T5_pooled'),('T8 pooled','terminal_T8_pooled'),('No sharing','network_no_sharing'),('Sharing, 0 min','network_sharing_0'),('Sharing, 40 min','network_sharing_40')]],
        'Service-resampling sensitivity at the original flight counts and four gates per terminal. The main baseline preserves each sampled day\'s original services and requests; only the sensitivity reassigns service values within airlines.')
    limitations=r'''
The original-data study uses only retained observed flights, with no added copies. Its main terminal and network experiments preserve each flight's matched service duration; by-airline empirical resampling is a separate sensitivity. Both use turnaround proxies rather than measured gate occupancy. Arrival is now the touchdown-time proxy (gate arrival minus TaxiIn), without a separate taxi-traversal stage. Simulated waiting therefore starts at this proxy and is not an observed gate-waiting measurement. There are no observed gate identifiers or complete gate trajectories. Tail-matched stays may include towing, and selected domestic carriers and same-day screening omit traffic. Capacities and compatibility are assumptions. Thus this case study does not establish actual JFK delays or an optimal airport gate count. Confidence intervals quantify simulation variation conditional on the selected inputs, excluding modeling and screening uncertainty.
'''
    s=replace_section(s,r'\section*{II.5\quad Interpretation and limitations}',r'\section*{II.6\quad Conclusion and reproducibility}',discussion('bts')+observed+limitations)
    conclusion=r'''
The main synthetic and BTS studies use the same original demand counts, service moments, terminal mapping, and four-gate capacity. Sharing and pooling are evaluated with paired inputs under these aligned assumptions. The following final experiment studies capacity and network size with separate, fully synthetic traffic; it does not expand the BTS records.

The \texttt{main\_synthetic\_4.json}, \texttt{main\_bts\_4.json}, and \texttt{main\_bts\_resampled\_4.json} configurations reproduce the synthetic study, observed-service BTS baseline, and resampling sensitivity with 100 replications and seed 553. The report's \texttt{results\_main} directory stores results, metadata, figures, and first-replication traces. Original cleaned data and provenance remain in \texttt{results\_aligned}. Superseded demand-expansion results are archived separately and are not part of the current main analysis.
'''
    s=replace_section(s,r'\section*{II.6\quad Conclusion and reproducibility}',r'\begin{thebibliography}{9}',conclusion)
    s=update_service_definition(s,reference)
    s=s.replace('2: Single terminal & T4 & 4 & DL flights on sampled date', '2: Single terminal & T4, T5, T8 separately & 4 each & Respective local flights')
    s=s.replace('Twenty automated tests pass','Twenty-one automated tests pass')
    replication_table=table(['Experiment','Independent replications'],[
        ['Calibrated synthetic main study','100 per policy'],['BTS main study','100 per policy'],
        ['Service-resampling sensitivity','100 per policy'],['Synthetic scaling grid','100 per configuration/policy'],
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
