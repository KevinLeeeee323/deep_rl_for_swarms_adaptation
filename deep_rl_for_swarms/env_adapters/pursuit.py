"""Thin adapter. Historical ma_envs sources and their stepping rules are unchanged."""
from contextlib import contextmanager
import numpy as np
from gym import spaces
from deep_rl_for_swarms.ma_envs.envs.point_envs.pursuit_evasion import PursuitEvasionEnv


def events(rewards, radius, timestep, limit=1024):
    """Use the original reward predicate, including its strict inequality."""
    return bool(rewards[0] > -1 / radius), bool(timestep >= limit)


class PursuitAdapter:
    def __init__(self, config, seed=0):
        self.config = dict(config)
        if config['nr_evaders'] != 1 or config['dynamics'] != 'unicycle':
            raise ValueError('This migration supports one scripted prey and unicycle dynamics.')
        if config['obs_mode'] not in ('sum_obs_no_ori', '2d_rbf'):
            raise ValueError('Use an explicitly supported observation mode.')
        self.raw = PursuitEvasionEnv(**config)
        self.rng = np.random.RandomState(seed)
        self.num_agents = config['nr_pursuers']
        self.observation_space = spaces.Box(-np.inf, np.inf,
            shape=self.raw.observation_space.shape, dtype=np.float32)
        self.action_space = self.raw.action_space
        self.state_dim = 4 * self.num_agents + 2
        self.ended = True

    @contextmanager
    def isolated_rng(self):
        # The legacy reset uses np.random globally. Isolate it without changing its draws.
        previous = np.random.get_state()
        np.random.set_state(self.rng.get_state())
        try:
            yield
        finally:
            self.rng.set_state(np.random.get_state())
            np.random.set_state(previous)

    def reset(self, initial=None):
        reset_world = self.raw.world.reset
        if initial is not None:
            pursuers = np.asarray(initial['pursuers'], dtype=np.float64)
            prey = np.asarray(initial['prey'], dtype=np.float64)
            if pursuers.shape != (self.num_agents, 3) or prey.shape != (1, 2):
                raise ValueError('Initial state shape mismatch')
            if not np.isfinite(pursuers).all() or not np.isfinite(prey).all():
                raise ValueError('Nonfinite initial state')
            def inject():
                self.raw.world.agent_states = pursuers.copy()
                self.raw.world.landmark_states = prey.copy()
                reset_world()
            self.raw.world.reset = inject
        try:
            with self.isolated_rng():
                obs = self.raw.reset()
        finally:
            self.raw.world.reset = reset_world
        self.ended = False
        return self._observations(obs), self.state()

    def initial(self):
        return dict(pursuers=self.raw.world.agent_states.tolist(),
                    prey=self.raw.world.landmark_states.tolist())

    def state(self):
        p = self.raw.world.agent_states
        features = np.column_stack((p[:, :2] / self.raw.world_size,
                                    np.sin(p[:, 2]), np.cos(p[:, 2])))
        return np.concatenate((features.ravel(),
            self.raw.world.landmark_states.ravel() / self.raw.world_size)).astype(np.float32)

    def distances(self):
        return self.raw.world.distance_matrix[-1, :self.num_agents].copy()

    def _observations(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.num_agents, *self.observation_space.shape) or not np.isfinite(obs).all():
            raise ValueError('Invalid observations')
        return obs

    def step(self, actions):
        if self.ended:
            raise RuntimeError('Reset before stepping an ended episode')
        actions = np.asarray(actions)
        if actions.shape != (self.num_agents, 2) or not np.isfinite(actions).all():
            raise ValueError('Invalid joint actions')
        with self.isolated_rng():
            obs, rewards, old_done, info = self.raw.step(actions)
        if not np.isfinite(rewards).all():
            raise ValueError('Nonfinite reward')
        terminated, truncated = events(rewards, self.raw.obs_radius,
                                       self.raw.timestep, self.raw.timestep_limit)
        assert bool(old_done) == (terminated or truncated)
        self.ended = terminated or truncated
        info = dict(info, success=terminated, old_done=bool(old_done),
                    min_distance=float(self.distances().min()))
        return self._observations(obs), self.state(), np.asarray(rewards), terminated, truncated, info

    def close(self):
        if self.raw.ax is not None:
            import matplotlib.pyplot as plt
            plt.close(self.raw.ax.figure)
