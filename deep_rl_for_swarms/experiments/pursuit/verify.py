"""Compare unchanged historical environment in an isolated process with the adapter."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import numpy as np
from .common import REFERENCE_COMMIT, save_json, provenance
from deep_rl_for_swarms.env_adapters.pursuit import PursuitAdapter

WORKER = r"""
import json, sys
from pathlib import Path
import numpy as np
from deep_rl_for_swarms.ma_envs.envs.point_envs.pursuit_evasion import PursuitEvasionEnv
import deep_rl_for_swarms.ma_envs.envs.point_envs.pursuit_evasion as original_module
assert Path(original_module.__file__).is_relative_to(Path(sys.argv[2]).parent)
cfg=json.load(open(sys.argv[1]))
seed=int(sys.argv[3]); mode=sys.argv[4]
np.random.seed(seed)
env=PursuitEvasionEnv(**cfg['environment'])
obs=env.reset()
rng=np.random.RandomState(seed+500)
trace=[]
for t in range(1024):
    if mode=='zero': a=np.zeros((env.nr_agents,2))
    elif mode=='boundary': a=rng.choice([-2.,-1.,0.,1.,2.],(env.nr_agents,2))
    else: a=rng.uniform(-1,1,(env.nr_agents,2))
    trace.append(dict(obs=np.asarray(obs),p=env.world.agent_states.copy(),e=env.world.landmark_states.copy(),
                      distances=env.world.distance_matrix[-1,:env.nr_agents].copy()))
    obs,r,done,info=env.step(a)
    trace[-1].update(reward=r.copy(),done=done)
    if done: break
trace.append(dict(obs=np.asarray(obs),p=env.world.agent_states.copy(),e=env.world.landmark_states.copy(),
                  distances=env.world.distance_matrix[-1,:env.nr_agents].copy()))
np.savez(sys.argv[2],**{k:np.array([x[k] for x in trace if k in x]) for k in trace[0]})
"""


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/pursuit/reference.json')
    p.add_argument('--output',default='runs/verification/environment.json')
    args=p.parse_args()
    config_path=Path(args.config).resolve()
    cfg=json.loads(config_path.read_text())
    repo=Path(__file__).resolve().parents[3]
    results=[]
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        archive=subprocess.check_output(['git','archive',REFERENCE_COMMIT,'deep_rl_for_swarms'],cwd=repo)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            for member in source.getmembers():
                if member.isfile():
                    target=root/member.name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(source.extractfile(member).read())
        (root/'worker.py').write_text(WORKER)
        for seed,mode in [(0,'random'),(1,'random'),(2,'random'),(3,'zero'),(4,'boundary')]:
            output=root/f'{seed}.npz'
            subprocess.run([sys.executable,str(root/'worker.py'),str(config_path),str(output),str(seed),mode],
                           cwd=root,env=dict(os.environ,PYTHONPATH=str(root),MPLBACKEND='Agg'),check=True)
            ref=np.load(output)
            env=PursuitAdapter(cfg['environment'],seed)
            obs,_=env.reset()
            rng=np.random.RandomState(seed+500)
            max_state,max_obs=0.,0.
            for t in range(len(ref['obs'])):
                np.testing.assert_allclose(env.raw.world.agent_states,ref['p'][t],rtol=1e-7,atol=1e-8)
                np.testing.assert_allclose(env.raw.world.landmark_states,ref['e'][t],rtol=1e-7,atol=1e-8)
                np.testing.assert_allclose(obs,ref['obs'][t],rtol=1e-7,atol=1e-7)
                np.testing.assert_allclose(env.distances(),ref['distances'][t],rtol=1e-7,atol=1e-8)
                max_state=max(max_state,float(np.max(np.abs(env.raw.world.agent_states-ref['p'][t]))),
                              float(np.max(np.abs(env.raw.world.landmark_states-ref['e'][t]))))
                max_obs=max(max_obs,float(np.max(np.abs(obs-ref['obs'][t]))))
                if t==len(ref['obs'])-1: break
                if mode=='zero': a=np.zeros((env.num_agents,2))
                elif mode=='boundary': a=rng.choice([-2.,-1.,0.,1.,2.],(env.num_agents,2))
                else: a=rng.uniform(-1,1,(env.num_agents,2))
                obs,_,r,term,trunc,_=env.step(a)
                np.testing.assert_allclose(r,ref['reward'][t],rtol=1e-7,atol=1e-8)
                assert (term or trunc)==bool(ref['done'][t])
            env.close()
            result=dict(seed=seed,mode=mode,steps=len(ref['reward']),max_state_error=max_state,max_obs_error=max_obs,status='PASS')
            results.append(result)
            print(json.dumps(result),flush=True)
    save_json(args.output,dict(status='PASS',reference_commit=REFERENCE_COMMIT,
        comparison='original source and adapter, SAME modern dependency stack; historical stack NOT tested',
        cases=results,provenance=provenance()))


if __name__=='__main__': main()
