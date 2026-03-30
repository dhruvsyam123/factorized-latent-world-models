from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from factor_latent_wm.config.core import DatasetConfig, EnvConfig
from factor_latent_wm.envs.multi_object_env import ACTION_DELTAS, MultiObjectEnv


def _move_towards(current: np.ndarray, target: np.ndarray) -> int:
    delta = target - current
    if np.abs(delta[0]) > np.abs(delta[1]):
        return 4 if delta[0] > 0 else 3
    if delta[1] != 0:
        return 2 if delta[1] > 0 else 1
    if delta[0] != 0:
        return 4 if delta[0] > 0 else 3
    return 0


def _goal_seek_policy(env: MultiObjectEnv) -> int:
    if env.current_task == "reach":
        return _move_towards(env.agent.position, env.target.position)
    if env.current_task == "key_door":
        if not env.has_key and env.key.active:
            return _move_towards(env.agent.position, env.key.position)
        if env.has_key and not env.door_open:
            return _move_towards(env.agent.position, env.door.position)
        return _move_towards(env.agent.position, env.target.position)
    if env.current_task == "push":
        if np.array_equal(env.block.position, env.target.position):
            return 0
        desired_push = env.target.position - env.block.position
        if np.abs(desired_push[0]) > np.abs(desired_push[1]):
            action = 4 if desired_push[0] > 0 else 3
        else:
            action = 2 if desired_push[1] > 0 else 1
        support_cell = env.block.position - ACTION_DELTAS[action]
        if np.array_equal(env.agent.position, support_cell):
            return action
        return _move_towards(env.agent.position, support_cell)
    return 0


def _push_helper_policy(env: MultiObjectEnv) -> int:
    if env.current_task == "push":
        return _goal_seek_policy(env)
    if env.current_task == "key_door" and not env.has_key:
        return _goal_seek_policy(env)
    return int(env.np_random.integers(0, len(ACTION_DELTAS)))


def _select_action(env: MultiObjectEnv, policy_name: str) -> int:
    if policy_name == "random":
        return int(env.np_random.integers(0, len(ACTION_DELTAS)))
    if policy_name == "goal_seek":
        return _goal_seek_policy(env)
    if policy_name == "push_helper":
        return _push_helper_policy(env)
    raise ValueError(f"Unknown policy {policy_name}")


def _generate_transitions(
    env_config: EnvConfig,
    num_transitions: int,
    policy_mix: tuple[str, ...],
    labelled: bool,
    seed: int,
) -> dict[str, np.ndarray]:
    env = MultiObjectEnv(env_config, seed=seed)
    rng = np.random.default_rng(seed)
    storage: dict[str, list[np.ndarray | int | float]] = defaultdict(list)

    obs, info = env.reset(seed=seed)
    while len(storage["actions"]) < num_transitions:
        policy_name = str(rng.choice(policy_mix))
        action = _select_action(env, policy_name)
        next_obs, _, terminated, truncated, next_info = env.step(action)

        storage["images"].append(obs.copy())
        storage["next_images"].append(next_obs.copy())
        storage["entity_features"].append(info["entity_features"].copy())
        storage["next_entity_features"].append(next_info["entity_features"].copy())
        storage["entity_mask"].append(info["entity_mask"].copy())
        storage["goal_vectors"].append(info["goal_vector"].copy())
        storage["task_ids"].append(info["task_id"])
        storage["actions"].append(action if labelled else -1)
        storage["labelled_mask"].append(1.0 if labelled else 0.0)

        if terminated or truncated:
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        else:
            obs, info = next_obs, next_info

    return {key: np.asarray(value) for key, value in storage.items()}


def save_dataset(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def generate_default_splits(
    output_dir: str | Path,
    env_config: EnvConfig | None = None,
    dataset_config: DatasetConfig | None = None,
) -> dict[str, Path]:
    env_config = env_config or EnvConfig()
    dataset_config = dataset_config or DatasetConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = {
        "passive_train.npz": (dataset_config.passive_train_transitions, False, dataset_config.seed),
        "passive_val.npz": (dataset_config.passive_val_transitions, False, dataset_config.seed + 1),
        "labelled_train.npz": (dataset_config.labelled_train_transitions, True, dataset_config.seed + 2),
        "labelled_val.npz": (dataset_config.labelled_val_transitions, True, dataset_config.seed + 3),
    }
    outputs: dict[str, Path] = {}
    for filename, (count, labelled, seed) in tqdm(split_specs.items(), desc="Generating splits"):
        arrays = _generate_transitions(env_config, count, dataset_config.policy_mix, labelled, seed)
        path = output_dir / filename
        save_dataset(path, arrays)
        outputs[filename] = path
    return outputs
