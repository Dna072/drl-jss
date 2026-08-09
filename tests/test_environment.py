"""Smoke tests for FactoryEnv contracts. Does not alter learning dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from custom_environment.environment_factory import init_custom_factory_env


@pytest.fixture()
def env():
    return init_custom_factory_env(
        is_verbose=False,
        max_steps=200,
        buffer_size=3,
        n_recipes=2,
        n_machines=2,
        is_evaluation=True,
    )


def test_reset_returns_dict_observation(env):
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    for key, value in obs.items():
        assert isinstance(value, np.ndarray), key
        assert value.ndim >= 1


def test_action_space_and_step(env):
    env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(next_obs, dict)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "JOBS_COMPLETED_ON_TIME" in info
    assert "JOBS_NOT_COMPLETED_ON_TIME" in info
    assert "CURRENT_TIME" in info


def test_episode_can_run_without_crash(env):
    env.reset()
    for _ in range(50):
        _obs, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    assert env.factory_time >= 0


def test_metrics_helpers(env):
    env.reset()
    assert env.get_tardiness_percentage() >= 0
    assert env.get_jobs_completed_on_time() >= 0
    assert env.get_jobs_completed_not_on_time() >= 0
