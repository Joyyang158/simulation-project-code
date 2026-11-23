"""Refresh the two-part paper from audited demand-study results and its saved template."""
from pathlib import Path
import csv
import shutil

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT.parent/'report'
SOURCE=REPORT/'archive_20gate_reference/report.tex'
RESULTS=ROOT/'results/stress'


def rows(mode, multiplier=5, filename='summary.csv'):
    return list(csv.DictReader((RESULTS/f'{mode}_20_x{multiplier}'/filename).open()))


def stat(mode,scenario,metric='mean_wait',multiplier=5):
    return next(r for r in rows(mode,multiplier) if r['scenario']==scenario and r['metric']==metric)


def number(mode,scenario,metric='mean_wait',multiplier=5):
    return float(stat(mode,scenario,metric,multiplier)['mean'])


def ci(r):
    return f"[{float(r['ci_low']):.2f}, {float(r['ci_high']):.2f}]"


def table(headers,data,caption):
    return ('\n\\begin{center}\n\\begin{minipage}{\\linewidth}\n\\centering\n'
        '\\captionof{table}{'+caption+'}\n\\small\n\\begin{tabular}{@{}'+'l'+'r'*(len(headers)-1)+'@{}}\n'
        '\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'+
        '\n'.join(' & '.join(row)+r' \\' for row in data)+
        '\n\\bottomrule\n\\end{tabular}\n\\end{minipage}\n\\end{center}\n')


def figure(path,caption,width='1'):
    return ('\n\\begin{center}\n\\begin{minipage}{\\linewidth}\n\\centering\n'
        '\\includegraphics[width='+width+r'\linewidth]{'+path+'}\n\\captionof{figure}{'+caption+'}\n'
        '\\end{minipage}\n\\end{center}\n')


def replace_section(text,start,end,body):
    a=text.index(start);b=text.index(end,a)
    return text[:a]+start+'\n'+body+'\n'+text[b:]


def results_section(mode):
    policies=[('Dedicated terminal','terminal_dedicated'),('Pooled terminal','terminal_pooled'),
        ('Smoothed terminal','terminal_smoothed'),('Network, no sharing','network_no_sharing'),
        ('Sharing, 0 min','network_sharing_0'),('Sharing, 10 min','network_sharing_10'),
        ('Sharing, 20 min','network_sharing_20'),('Sharing, 40 min','network_sharing_40'),
        ('One gate closed','network_closed_gate'),('Initially occupied','network_initial_occupied')]
    data=[[label,f"{number(mode,key):.2f}",ci(stat(mode,key))] for label,key in policies]
    out=table(['Policy','Mean wait (min)',r'95\% CI'],data,
        'Main fivefold-demand stress test: 100 replications, 20 gates per terminal. Additional flights are generated, not additional observations.')
    out+=figure(f'results_stress/{mode}_20_x5/experiment_overview.pdf',
        'Main stress-test outcomes. Demand scaling applies to terminal and network settings; the single-gate panel retains its independently specified offered loads.')
    data=[]
    for rho in [.5,.7,.85,.95]:
        a=f'single_{rho:g}_base';b=f'single_{rho:g}_low_variability'
        d=next(r for r in rows(mode,filename='paired_differences.csv') if r['alternative']==b and r['metric']=='mean_wait')
        data.append([f'{rho:.2f}',f'{number(mode,a):.2f}',f'{number(mode,b):.2f}',f"{float(d['mean']):.2f}",ci(d)])
    out+=table([r'$\rho$','Original','Lower variability','Difference',r'95\% paired CI'],data,
        'Single-gate mean waiting, minutes. These controls are identical across demand multipliers within each input mode.')
    out+=figure(f'results_stress/{mode}_20_x5/policy_differences.pdf',
        r'Fivefold demand: paired mean-wait differences (alternative minus baseline); pointwise 95\% confidence intervals.','.8')
    data=[]
    for m in [1,4,5,6]:
        data.append([f'{m}x',f'{125.064516129*m:.2f}',f'{number(mode,"terminal_pooled",multiplier=m):.2f}',
            f'{number(mode,"network_no_sharing",multiplier=m):.2f}',
            f'{number(mode,"network_sharing_0",multiplier=m):.2f}',
            f'{number(mode,"network_sharing_40",multiplier=m):.2f}'])
    out+=table(['Demand','Ref. flights/day','T4 pooled','No sharing','Share 0 min','Share 40 min'],data,
        'Demand sensitivity: mean waits in minutes. Reference daily counts use all 31 dates; the simulation bootstrap has a slightly different average.')
    out+=figure(f'results_stress/{mode}_demand_sensitivity.pdf',
        r'Demand sensitivity with unchanged gate capacity. Error bars show pointwise 95\% confidence intervals over 100 replications at each level.')
    return out


def discussion(mode):
    n=number(mode,'network_no_sharing');zero=number(mode,'network_sharing_0');forty=number(mode,'network_sharing_40')
    d=next(r for r in rows(mode,filename='paired_differences.csv') if r['alternative']=='network_sharing_0' and r['metric']=='mean_wait')
    return (f'At the reference demand (1x), pooled and network mean waiting remains zero in the sampled runs. '
        f'At fivefold demand, network mean waiting is {n:.2f} minutes without sharing, '
        f'{zero:.2f} with zero-transfer sharing, and {forty:.2f} with a 40-minute transfer. '
        f'The paired zero-transfer effect is {float(d["mean"]):.2f} minutes, with 95\\% CI {ci(d)}. '
        f'T4 pooled waiting is {number(mode,"terminal_pooled"):.2f} minutes; smoothing gives '
        f'{number(mode,"terminal_smoothed"):.2f} minutes. These are simulated policy outcomes under the declared demand scenario.\n\n'
        'Mean network offered load alone does not describe peak congestion: the three terminals have unequal demand, '
        'and arrivals concentrate in particular periods. At 5x, T5 has offered load above one even though the network mean is below one. '
        'Transfers consume gate capacity during reservation as well as service. The 4x and 6x results show how policy effects change with demand; '
        'they are finite-day experiments with a drain, not proofs of steady-state stability. '
        'The single-gate service-variability comparison is unchanged by this demand expansion.\n')


def main():
    assert (RESULTS/'audit.json').exists(), 'Run summarize_stress_study.py first'
    shutil.copytree(RESULTS,REPORT/'results_stress',dirs_exist_ok=True)
    text=SOURCE.read_text()
    syn=number('synthetic','network_no_sharing'); syn0=number('synthetic','network_sharing_0')
    real=number('bts','network_no_sharing');real0=number('bts','network_sharing_0')
    abstract=(r'We compare calibrated synthetic inputs and BTS-based demand stress tests using T4, T5, and T8, with 20 assumed gates per terminal and ten shared-compatible gates per terminal. '
        'The reference data contain 3,877 screened JFK domestic turnarounds across 31 days in July 2025. '
        'To examine congestion at this capacity, both input models use the same 1x reference demand and 4x, 5x, and 6x constructed demand levels. '
        'The main 5x scenario has 625.32 modeled flights per day averaged across the 31 reference dates and has mean network offered load of 76.5\\%. '
        'Each policy uses 100 independent replications, common random numbers, and paired confidence intervals. '
        f'At 5x, zero-transfer sharing changes mean network waiting from {syn:.2f} to {syn0:.2f} minutes with synthetic inputs '
        f'and from {real:.2f} to {real0:.2f} minutes with BTS-based inputs. '
        'The 1x baseline retains zero pooled network waiting in the sampled runs. Additional demand is generated from the reference sample; '
        'it is not additional measured BTS traffic. Capacities, compatibility, and continuous gate occupancy remain assumptions.\n\n'
        r'\textbf{Keywords:} discrete-event simulation; gate queues; demand stress tests; BTS; common random numbers.'+'\n')
    text=replace_section(text,r'\section*{Abstract}',r'\section*{Common experimental settings}',abstract)
    text=text.replace('Using full offered service demand', 'At the 1x reference demand, using full offered service demand')
    text=text.replace('The same settings apply to both parts. Only the stochastic input construction differs.',
        'The same capacities, demand multipliers, and policy settings apply to both parts. The stochastic input construction differs.')
    text=text.replace('DL flights on sampled date',r'$m\times$ sampled DL count').replace('All selected flights on sampled date',r'$m\times$ sampled network count')
    marker=r'\part*{Part I: Calibrated Synthetic Experiments}'
    demand=r'''
\section*{Demand scaling and scenario rationale}
The primary experiment uses $m=5$ times the retained daily flight counts, while the 1x reference is kept as a low-demand baseline and 4x and 6x are sensitivity levels. Raising capacity from four to twenty gates is a fivefold increase; choosing fivefold demand restores the previous BTS average demand-to-capacity ratio without assuming the old waiting results remain valid. This design is based on offered workload, not on selecting a multiplier after observing a desired policy outcome.

For each sampled day, both parts use exactly $m$ times its flight count for each airline. In the BTS-based model, original requests keep their observed times. Each additional flight receives a new identifier and an independent uniform arrival time inside its source flight's 30-minute bin. Thus each day's airline-by-bin counts scale exactly, without forcing identical arrival timestamps. Service durations are sampled independently from the airline's empirical service pool. Synthetic runs use the same expanded counts and carrier mapping, but independently generate all arrival times from the calibrated airline histogram and all services from the moment-matched Gamma distribution. Policies share the same pre-generated inputs within each replication. A separate expansion stream leaves date selection unchanged.

At 5x, reference mean daily counts are 232.74 for T4, 281.29 for T5, and 111.29 for T8, totaling 625.32. The 100 sampled days average 621.45 generated flights. The cleaned dataset still contains 3,877 real visits; generated flight counts are scenario sizes, not newly collected data. The one-gate experiment is not multiplied: its Poisson arrivals remain controlled by the specified $\rho$.
'''
    demand+=table(['Multiplier','Ref. flights/day','T4 load','T5 load','T8 load','Network load'],[
        [f'{m}x',f'{125.064516129*m:.2f}',f'{17.25739247*m:.1f}\\%',f'{20.37668*m:.1f}\\%',f'{8.254368*m:.1f}\\%',f'{15.29614682*m:.1f}\\%'] for m in [1,4,5,6]],
        r'Offered service workload divided by full-day gate capacity. Values above 100\% indicate more offered workload than nominal full-day local capacity; in-window utilization is a separate metric.')
    text=text.replace(marker,demand+'\n'+marker)
    text=text.replace('Preserve its number of flights for each carrier and its airline-to-terminal mapping.',
        'Multiply its number of flights for each carrier by the common demand multiplier and preserve its airline-to-terminal mapping.')
    text=text.replace('No extra flights are added to make the 20-gate scenarios congested.',
        'The demand multipliers apply only to the multi-gate settings; single-gate random inputs remain identical across these multipliers.')
    text=text.replace('Seventeen automated tests pass','Nineteen automated tests pass')
    text=text.replace('and positive synthetic services.',
        'and positive synthetic services. Additional tests verify scaled airline-by-bin counts, unique identifiers, retained originals, nested expansion streams, identical cross-mode demand counts, and unchanged single-gate controls.')
    text=replace_section(text,r'\section*{I.5\quad Results and interpretation}',r'\section*{I.6\quad Discussion}',results_section('synthetic'))
    text=replace_section(text,r'\section*{I.6\quad Discussion}',r'\clearpage',discussion('synthetic').split('\n\n')[0]+
        '\n\nThe sensitivity figure shows the role of demand at fixed capacity. These finite-day comparisons allow post-day draining and do not establish steady-state stability. Gamma tail and arrival-dependence limitations are described in the input construction.\n')
    text=replace_section(text,r'\section*{II.1\quad Purpose and alignment}',r'\section*{II.2\quad Real data, screening, and input diagnostics}',
        'This part uses real BTS records as reference inputs for constructed demand scenarios. The 1x baseline uses observed request times. '
        'The main 5x experiment and 4x/6x sensitivities add generated requests in the source arrival bins. All capacity, airline mapping, demand multiplier, service-moment, policy, seed, and horizon settings match Part I. '
        'The added flights are not observed historical flights, and results are not estimates of measured JFK gate delays.\n')
    text=text.replace('Accordingly, we report a by-airline empirical-resampling baseline and a second mode that replays each selected day\'s observed service values together with its arrivals.',
        'Accordingly, we report by-airline empirical resampling and a source-service sensitivity. At 1x this second mode replays the selected day; at 5x added flights inherit their source service values while receiving generated arrival times, so the expanded series is not a fully observed dependence-preserving replay.')
    text=replace_section(text,r'\section*{II.3\quad Simulation design}',r'\section*{II.4\quad Results}',r'''
The event engine, assignment rules, metrics, and paired confidence intervals are those in Part I. Each replication samples a full reference date. At demand multiplier $m$, each original flight supplies $m-1$ additional flights as described in the common design. Every service in the primary model is sampled independently from its airline's July pool. The source-service sensitivity at 5x retains each original service and copies it to added flights. Originals retain their time-service pairing; within-bin jitter changes the ordering of generated flights. This sensitivity tests a different service assignment, not a reconstruction of unobserved real traffic.

All scenarios retain twenty gates and ten shared-compatible gates per terminal. The single-gate experiment still uses synthetic Poisson arrivals and pooled empirical service with mean 105.67 minutes, matched to Part I's mean. The same seed and replication index pair daily demand across input modes; policy confidence intervals compare policies within an input mode.
''')
    text=replace_section(text,r'\section*{II.4\quad Results}',r'\section*{II.5\quad Interpretation and limitations}',results_section('bts'))
    sensitivity=list(csv.DictReader((RESULTS/'bts_20_x5_observed/summary.csv').open()))
    sens={(r['scenario'],r['metric']):r for r in sensitivity}
    data=[]
    for label,key in [('T4 pooled','terminal_pooled'),('No sharing','network_no_sharing'),('Sharing 0 min','network_sharing_0'),('Sharing 10 min','network_sharing_10'),('Sharing 20 min','network_sharing_20'),('Sharing 40 min','network_sharing_40')]:
        r=sens[key,'mean_wait']
        data.append([label,f'{number("bts",key):.2f}',f"{float(r['mean']):.2f}",ci(r)])
    interpret=discussion('bts')+table(['Policy','Resampled service','Source service',r'Source 95\% CI'],data,
        'Fivefold BTS-based demand: sensitivity to service assignment, mean waits in minutes. Both columns include constructed additional flights.')
    interpret+=r'''
A transfer-time threshold is not inferred by extrapolation beyond the tested 0--40 minute grid. Source-service sensitivity changes only how services are assigned; it does not recover missing gate histories. Confidence intervals quantify replication variation conditional on each constructed scenario and omit uncertainty in data screening and model assumptions.

Real source records contain no observed gate IDs or occupancy trajectories. Tail-matched turnaround may include towing or off-gate time. Selected carriers, domestic coverage, overnight exclusions, and the stay cutoff omit substantial demand. Fivefold expansion is a declared stress scenario, not a claim that omitted real traffic is exactly five times larger or shares these airlines' characteristics. Capacities and compatibility remain hypothetical. Synthetic calibration uses the full reference sample and is not out-of-sample prediction validation. At overloaded terminals, finite-day draining permits all flights to finish after the observation horizon; it does not establish steady-state stability.
'''
    text=replace_section(text,r'\section*{II.5\quad Interpretation and limitations}',r'\section*{II.6\quad Conclusion and reproducibility}',interpret)
    conclusion=(f'The revised study varies demand while fixing capacity at twenty gates per terminal. The 1x baseline demonstrates spare capacity for the retained subset; '
        f'the primary 5x stress test produces network waiting of {syn:.2f} minutes in the synthetic model and {real:.2f} minutes in the BTS-based model before sharing. '
        f'Zero-transfer sharing gives {syn0:.2f} and {real0:.2f} minutes, respectively. '
        'The 4x/6x sensitivities, paired policy intervals, and source-service comparison show how conclusions depend on demand and input construction. These results describe declared scenarios rather than actual airport delays.\n\n'+r'''
The code scripts \texttt{run\_stress\_study.py}, \texttt{summarize\_stress\_study.py}, and \texttt{build\_stress\_report.py} reproduce the runs, audit demand alignment, generate figures, and refresh the paper. Each scenario uses the same cleaned July input, 100 replications, and seed 553. The report's \texttt{results\_stress} directory contains summaries, paired differences, metadata, first-replication records, and figures. Source data and cleaning provenance remain in \texttt{results\_aligned}. Earlier reference-only and four-gate PDFs are archived separately. The original template, plain Abstract, and page numbers are retained.
''')
    text=replace_section(text,r'\section*{II.6\quad Conclusion and reproducibility}',r'\begin{thebibliography}{9}',conclusion)
    text = text.replace('$m=5$', '$M=5$').replace('$m$', '$M$').replace('$m-1$', '$M-1$').replace(r'$m\times$', r'$M\times$')
    (REPORT/'report.tex').write_text(text)
    first=text.index(r'\part*{Part I:');second=text.index(r'\part*{Part II:');bib=text.index(r'\begin{thebibliography}')
    common=text[:first];references=text[bib:]
    (REPORT/'report_synthetic.tex').write_text(common+text[first:second].replace('\\clearpage\n','')+references)
    (REPORT/'report_bts.tex').write_text(common+text[second:bib]+references)
    print('Updated main and standalone reports with fivefold main demand and 1x/4x/6x sensitivity.')


if __name__=='__main__':main()
