"""Plot saved validation curves and independent evaluation results."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--evaluations',nargs='*',default=[])
    p.add_argument('--output',required=True)
    a=p.parse_args()
    root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    series=[]
    for run in a.runs:
        rows=[json.loads(line) for line in (Path(run)/'metrics.jsonl').read_text().splitlines()]
        series.append([r for r in rows if 'validation' in r])
    common=sorted(set.intersection(*[set(r['steps'] for r in rows) for rows in series]))
    if not common: raise ValueError('No common evaluation budgets')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key,label in zip(axes,['success_rate','raw_return'],['Validation capture rate','Validation raw return']):
        values=np.array([[next(r['validation'][key] for r in rows if r['steps']==step)
                          for step in common] for rows in series])
        mean=values.mean(0)
        ax.plot(common,mean,marker='o',label=f'{len(series)} training seed(s)')
        if len(series)>1:
            std=values.std(0,ddof=1)
            ax.fill_between(common,mean-std,mean+std,alpha=.2,label='between-seed sample SD')
        ax.set(xlabel='Environment steps',ylabel=label)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_ylim(0,1)
    fig.savefig(root/'learning.png',dpi=180)
    plt.close(fig)
    table=[]
    for location in a.evaluations:
        result=json.loads((Path(location)/'results.json').read_text())
        table.append(dict(name=Path(location).name,**result['summary'],suite_hash=result['suite_hash']))
    if table:
        if len({row['suite_hash'] for row in table})!=1:
            raise ValueError('Do not compare different test suites')
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        ax.bar([r['name'] for r in table],[r['success_rate'] for r in table],color='tab:blue')
        ax.set(ylabel='Test capture rate',ylim=(0,1),title='Same fixed initial states; individual models')
        ax.tick_params(axis='x',labelrotation=20)
        fig.savefig(root/'evaluation.png',dpi=180)
        plt.close(fig)
        (root/'table.json').write_text(json.dumps(table,indent=2)+'\n')
    (root/'plot-manifest.json').write_text(json.dumps(dict(runs=a.runs,evaluations=a.evaluations,
        common_steps=common,smoothing='none',band='between-training-seed sample SD if n>1'),indent=2))


if __name__=='__main__': main()
