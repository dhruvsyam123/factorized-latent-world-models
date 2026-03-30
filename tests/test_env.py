import numpy as np

from factor_latent_wm.config.core import EnvConfig
from factor_latent_wm.envs import ENTITY_FEATURE_DIM, MAX_ENTITIES, MultiObjectEnv


def test_env_reset_and_step_shapes():
    env = MultiObjectEnv(EnvConfig(render_size=84, max_hazards=1, max_distractors=1), seed=3)
    obs, info = env.reset(seed=3, options={"task": "reach"})
    assert obs.shape == (84, 84, 3)
    assert info["entity_features"].shape == (MAX_ENTITIES, ENTITY_FEATURE_DIM)
    assert info["entity_mask"].shape == (MAX_ENTITIES,)
    assert info["state_tokens"].shape == (MAX_ENTITIES, ENTITY_FEATURE_DIM)

    next_obs, reward, terminated, truncated, next_info = env.step(4)
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert next_info["task"] == "reach"


def test_push_task_moves_block_when_agent_pushes():
    env = MultiObjectEnv(EnvConfig(max_hazards=0, max_distractors=0), seed=0)
    env.reset(seed=0, options={"task": "push"})
    env.agent.position = np.array([2, 2])
    env.block.position = np.array([3, 2])
    env.target.position = np.array([5, 2])

    env.step(4)
    assert np.array_equal(env.agent.position, np.array([3, 2]))
    assert np.array_equal(env.block.position, np.array([4, 2]))


def test_task_specific_reward_shaping():
    env = MultiObjectEnv(EnvConfig(max_hazards=0, max_distractors=0), seed=1)
    env.reset(seed=1, options={"task": "push"})
    env.agent.position = np.array([1, 1])
    env.block.position = np.array([2, 1])
    env.target.position = np.array([6, 1])
    far_reward = env._reward()
    env.block.position = np.array([5, 1])
    near_reward = env._reward()
    assert near_reward > far_reward
