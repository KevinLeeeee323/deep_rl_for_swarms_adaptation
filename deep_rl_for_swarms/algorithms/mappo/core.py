"""Shared continuous Actor, team-state Critic and MAPPO update."""
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(inputs, outputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, outputs))


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    rewards, values, next_values = map(np.asarray, (rewards, values, next_values))
    advantage = np.zeros_like(rewards, dtype=np.float32)
    carry = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (not terminated[t]) - values[t]
        carry = delta + gamma * lam * (not (terminated[t] or truncated[t])) * carry
        advantage[t] = carry
    return advantage, advantage + values


class MAPPO(nn.Module):
    def __init__(self, obs_dim, state_dim, config, device='cpu'):
        super().__init__()
        self.config = dict(config)
        self.actor = mlp(obs_dim, 2, config['hidden'])
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        self.critic = mlp(state_dim, 1, config['hidden'])
        # Small initial means; critic retains normal initialization.
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['learning_rate'])

    @property
    def device(self):
        return next(self.parameters()).device

    def tensor(self, data):
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(-5, 1).exp())

    def log_prob(self, obs, latent):
        # Stable Jacobian for a=tanh(z); retain z in the rollout to avoid atanh clipping.
        dist = self.distribution(obs)
        correction = 2 * (math.log(2) - latent - torch.nn.functional.softplus(-2 * latent))
        return (dist.log_prob(latent) - correction).sum(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs = self.tensor(obs)
        dist = self.distribution(obs)
        latent = dist.mean if deterministic else dist.sample()
        return (latent.tanh().cpu().numpy(), latent.cpu().numpy(),
                self.log_prob(obs, latent).cpu().numpy())

    @torch.no_grad()
    def value(self, state):
        return self.critic(self.tensor(state)).squeeze(-1).cpu().numpy()

    def update(self, rollout):
        cfg = self.config
        adv, returns = gae(rollout['rewards'], rollout['values'], rollout['next_values'],
                          rollout['terminated'], rollout['truncated'], cfg['gamma'], cfg['gae_lambda'])
        obs, latent, old_lp, states = [self.tensor(np.asarray(rollout[k]))
                                      for k in ('obs', 'latent', 'log_prob', 'states')]
        advantages = self.tensor((adv - adv.mean()) / (adv.std() + 1e-8))
        returns = self.tensor(returns)
        nsteps, nagents = old_lp.shape
        metrics = []
        # Minibatches index world transitions, retain all agents within each transition.
        for _ in range(cfg['epochs']):
            permutation = torch.randperm(nsteps, device=self.device)
            for idx in permutation.split(cfg['minibatch']):
                logp = self.log_prob(obs[idx], latent[idx])
                ratio = (logp - old_lp[idx]).exp()
                a = advantages[idx, None]
                actor_loss = -torch.minimum(ratio * a, ratio.clamp(1-cfg['clip'], 1+cfg['clip']) * a).mean()
                value_loss = (self.critic(states[idx]).squeeze(-1) - returns[idx]).square().mean()
                # Fresh reparameterized samples estimate entropy of the bounded policy.
                fresh = self.distribution(obs[idx]).rsample()
                entropy = -self.log_prob(obs[idx], fresh).mean()
                loss = actor_loss + cfg['value_coef'] * value_loss - cfg['entropy_coef'] * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite MAPPO loss')
                self.optimizer.zero_grad()
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.parameters(), cfg['max_grad_norm'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite MAPPO gradient')
                self.optimizer.step()
                metrics.append([actor_loss.item(), value_loss.item(), entropy.item(),
                                ((ratio - 1) - (logp - old_lp[idx])).mean().item()])
        return dict(zip(('actor_loss','value_loss','entropy','approx_kl'), np.mean(metrics, axis=0).tolist()))
