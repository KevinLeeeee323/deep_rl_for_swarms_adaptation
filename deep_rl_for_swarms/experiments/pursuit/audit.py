"""Independently recompute metrics from saved positions, not the evaluator's distances."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import save_json, digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite',required=True)
    p.add_argument('--evaluations',nargs='+',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    suite=json.loads(Path(a.suite).read_text())
    checked=[]
    for location in a.evaluations:
        root=Path(location)
        result=json.loads((root/'results.json').read_text())
        assert result['suite_hash']==digest(suite)
        assert result['config']['environment']==suite['environment']
        cfg=suite['environment']; radius=cfg['comm_radius']/2
        assert len(result['episodes'])==len(suite['scenarios'])
        successes=[]; steps=[]; returns=[]; maximum=0.
        for i,(row,scene) in enumerate(zip(result['episodes'],suite['scenarios'])):
            trace=np.load(root/f'trace-{i:04d}.npz')
            np.testing.assert_array_equal(trace['pursuers'][0],scene['pursuers'])
            np.testing.assert_array_equal(trace['prey'][0],scene['prey'])
            assert row['scene_id']==scene['id']
            delta=np.abs(trace['pursuers'][:,:,:2]-trace['prey'][:,0,None,:])
            if cfg['torus']: delta=np.where(delta>cfg['world_size']/2,cfg['world_size']-delta,delta)
            distances=np.sqrt(np.sum(delta**2,axis=-1))
            np.testing.assert_allclose(distances,trace['distances'],rtol=1e-7,atol=1e-8)
            rewards=-np.minimum(distances[1:].min(axis=1),radius)/radius
            raw_return=float(rewards.sum())
            maximum=max(maximum,abs(raw_return-row['raw_return']))
            assert abs(raw_return-row['raw_return'])<1e-8
            n=len(rewards); captured=bool(rewards[-1]>-1/radius)
            assert row['success']==captured and row['terminated']==captured
            assert row['truncated']==(n>=1024) and row['steps']==n
            assert n==1024 or captured
            successes.append(captured); steps.append(n); returns.append(raw_return)
        conditional=[n for n,s in zip(steps,successes) if s]
        expected=dict(episodes=len(steps),success_rate=float(np.mean(successes)),
            capture_steps=float(np.mean(conditional)) if conditional else None,
            penalized_steps=float(np.mean([n if s else 1024 for n,s in zip(steps,successes)])),
            raw_return=float(np.mean(returns)))
        for key,value in expected.items():
            if value is None: assert result['summary'][key] is None
            else: assert abs(result['summary'][key]-value)<1e-8
        checked.append(dict(evaluation=location,episodes=len(steps),
            max_return_error=maximum,status='PASS',summary=expected))
    save_json(a.output,dict(status='PASS',suite_hash=digest(suite),checks=checked))
    print(json.dumps(checked,indent=2))


if __name__=='__main__': main()
