import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from deep_rl_for_swarms.env_adapters.pursuit import PursuitAdapter, events
from deep_rl_for_swarms.algorithms.mappo.core import MAPPO, gae
from deep_rl_for_swarms.experiments.pursuit.common import evaluate, scenarios, summarize, checkpoint, load_model

CONFIG=json.loads((Path(__file__).resolve().parents[1]/'configs/pursuit/reference.json').read_text())


@pytest.mark.parametrize('d,success',[(.999999,True),(1.,False),(1.000001,False),(200.,False)])
@pytest.mark.parametrize('t,timeout',[(1023,False),(1024,True)])
def test_capture_timeout(d,success,t,timeout):
    R=CONFIG['environment']['comm_radius']/2
    assert events([-min(d,R)/R],R,t)==(success,timeout)


def test_reset_reproducible_and_no_global_rng_pollution():
    a=PursuitAdapter(CONFIG['environment'],42)
    b=PursuitAdapter(CONFIG['environment'],42)
    np.random.seed(7); expected=np.random.rand(4)
    np.random.seed(7); obs,state=a.reset()
    np.testing.assert_equal(np.random.rand(4),expected)
    other,_=b.reset()
    np.testing.assert_equal(obs,other)
    initial=a.initial()
    action=np.full((10,2),.2)
    first=a.step(action)
    a.reset(initial)
    second=a.step(action)
    np.testing.assert_equal(first[0],second[0])
    np.testing.assert_equal(first[1],second[1])
    assert obs.shape==(10,503) and state.shape==(42,)
    assert obs.min()<0  # Do not clip signed angular features to the old [0,1] metadata.
    assert not any(k.startswith(('tensorflow','mpi4py')) for k in sys.modules)
    a.close(); b.close()


def test_periodic_distance_and_reward():
    e=PursuitAdapter(CONFIG['environment'],6); e.reset()
    initial=e.initial()
    initial['pursuers'][0]=[.2,20,0]
    initial['prey']=[[99.8,20]]
    e.reset(initial)
    assert e.distances()[0]==pytest.approx(.4)
    expected=-min(e.distances().min(),e.raw.obs_radius)/e.raw.obs_radius
    np.testing.assert_allclose(e.raw.get_reward(np.zeros((10,2))),np.full(10,expected))
    e.close()


def test_timeout_bootstrap_and_no_cross_episode_gae():
    adv,ret=gae([1.,2.,3.],[0.,0.,0.],[10.,10.,10.],
                [False,True,False],[True,False,False],.9,1.)
    np.testing.assert_allclose(adv,[10.,2.,12.])
    # A timeout bootstraps but never imports the next episode's advantage.
    adv,_=gae([1.,2.],[0.,0.],[5.,7.],[False,False],[False,False],.9,1.)
    np.testing.assert_allclose(adv,[1+4.5+.9*(2+6.3),2+6.3])


def test_policy_transform_gradient_and_checkpoint(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(2)
    e=PursuitAdapter(CONFIG['environment']); obs,state=e.reset()
    m=MAPPO(503,42,dict(CONFIG['algorithm'],epochs=1,minibatch=4))
    latent=m.tensor(np.full((10,2),.3)); latent.requires_grad_(True)
    logp=m.log_prob(m.tensor(obs),latent)
    assert torch.isfinite(logp).all()
    z=latent.detach(); dist=m.distribution(m.tensor(obs))
    manual=(dist.log_prob(z)-torch.log(1-z.tanh().square())).sum(-1)
    torch.testing.assert_close(logp,manual)
    logp.sum().backward(); assert torch.isfinite(latent.grad).all()
    data={k:[] for k in ('obs','states','latent','log_prob','rewards','values','next_values','terminated','truncated')}
    before=[p.detach().clone() for p in m.parameters()]
    for _ in range(8):
        action,z,lp=m.act(obs); value=float(m.value(state))
        no,ns,r,term,trunc,_=e.step(action)
        row=dict(obs=obs.copy(),states=state.copy(),latent=z,log_prob=lp,rewards=float(r[0]),
                 values=value,next_values=float(m.value(ns)),terminated=term,truncated=trunc)
        for k,v in row.items(): data[k].append(v)
        obs,state=e.reset() if term or trunc else (no,ns)
    metrics=m.update(data)
    assert np.isfinite(list(metrics.values())).all()
    assert any(not torch.equal(a,b) for a,b in zip(before,m.parameters()))
    path=tmp_path/'model.pt'
    cfg=dict(CONFIG,algorithm=m.config)
    checkpoint(path,m,cfg,8,2,'test-hash')
    loaded,_=load_model(path)
    np.testing.assert_equal(m.act(obs,True)[0],loaded.act(obs,True)[0])
    e.close()


def test_metrics_and_evaluation_do_not_modify_training_env():
    rows=[dict(success=True,steps=10,raw_return=-2.),dict(success=False,steps=1024,raw_return=-100.)]
    assert summarize(rows)==dict(episodes=2,success_rate=.5,capture_steps=10.,penalized_steps=517.,raw_return=-51.)
    assert summarize([rows[1]])['capture_steps'] is None
    env=PursuitAdapter(CONFIG['environment']); obs,state=env.reset()
    before=env.initial()
    m=MAPPO(503,42,CONFIG['algorithm'])
    weights={k:v.clone() for k,v in m.state_dict().items()}
    suite=scenarios(CONFIG,1,100000,'validation')
    # Use a valid near-capture initial state; do not alter prey's script.
    suite['scenarios'][0]['pursuers'][0]=[11.2,12.3,.4]
    suite['scenarios'][0]['prey']=[[11.21,12.31]]
    evaluate(CONFIG,suite,m)
    assert env.initial()==before
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in weights.items())
    env.close()


@pytest.mark.parametrize('distance,success',[(.999999,True),(1.,False),(1.000001,False)])
@pytest.mark.parametrize('step,timeout',[(0,False),(1023,True)])
def test_real_step_capture_and_timeout(distance,success,step,timeout):
    # Freeze motion only in this unit fixture to isolate the original post-step event rule.
    # Production uses the untouched scripted prey.
    e=PursuitAdapter(CONFIG['environment'],17); e.reset()
    initial=e.initial()
    initial['prey']=[[12.3,14.7]]
    initial['pursuers'][0]=[12.3+distance,14.7,.2]
    # Set y distance for exact representability near threshold.
    initial['prey']=[[12.,14.]]
    initial['pursuers'][0]=[12.+distance,14.,.2]
    e.reset(initial)
    e.raw.world.scripted_agents[0].action_callback=lambda agent,world: np.zeros(2)
    e.raw.timestep=step
    _,_,_,term,trunc,info=e.step(np.zeros((10,2)))
    assert term==success and trunc==timeout
    assert info['old_done']==(success or timeout)
    if term or trunc:
        with pytest.raises(RuntimeError): e.step(np.zeros((10,2)))
    e.close()


def test_original_render_does_not_advance_world():
    e=PursuitAdapter(CONFIG['environment'],8); e.reset()
    state=e.initial()
    e.raw.render()
    assert e.initial()==state and e.raw.timestep==0
    e.close()


def test_preexisting_exact_midline_error_is_not_silently_hidden():
    e=PursuitAdapter(CONFIG['environment'],8); e.reset()
    initial=e.initial(); initial['prey']=[[50.,25.]]
    e.reset(initial)
    with pytest.raises(UnboundLocalError): e.step(np.zeros((10,2)))
    e.close()
