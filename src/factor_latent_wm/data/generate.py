from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from factor_latent_wm.config.core import DatasetConfig, EnvConfig
from factor_latent_wm.envs.multi_object_env import ACTION_DELTAS, MultiObjectEnv, SPLIT_IDS, TASK_FAMILIES


COLLECTION_MODE_IDS = {
    "passive": 0,
    "labelled": 1,
}


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


def _pad_sequence(sequence: list[np.ndarray], target_length: int) -> list[np.ndarray]:
    if not sequence:
        raise ValueError("Cannot pad empty sequence")
    padded = list(sequence)
    while len(padded) < target_length:
        padded.append(padded[-1].copy())
    return padded


def _pad_vector_sequence(sequence: list[np.ndarray], target_length: int) -> list[np.ndarray]:
    return _pad_sequence(sequence, target_length)


def _collect_episode(
    env_config: EnvConfig,
    policy_mix: tuple[str, ...],
    labelled: bool,
    seed: int,
    split_name: str,
) -> dict[str, np.ndarray]:
    env = MultiObjectEnv(env_config, seed=seed)
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed, options={"split": split_name})

    frames = [obs.copy()]
    state_tokens = [info["state_tokens"].copy()]
    state_masks = [info["state_mask"].copy()]
    goal_vectors = [info["goal_vector"].copy()]
    task_ids = [int(info["task_id"])]
    task_family_ids = [int(info["task_family_id"])]
    split_ids = [int(info["split_id"])]
    collection_mode_ids = [COLLECTION_MODE_IDS["labelled" if labelled else "passive"]]
    task_progress = [float(info["task_progress"])]
    actions = []
    rewards = []
    terminals = []
    truncateds = []
    action_mask = []

    for _ in range(env_config.max_steps):
        policy_name = str(rng.choice(policy_mix))
        action = _select_action(env, policy_name)
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        actions.append(action if labelled else -1)
        rewards.append(float(reward))
        terminals.append(float(terminated))
        truncateds.append(float(truncated))
        action_mask.append(1.0)

        frames.append(next_obs.copy())
        state_tokens.append(next_info["state_tokens"].copy())
        state_masks.append(next_info["state_mask"].copy())
        goal_vectors.append(next_info["goal_vector"].copy())
        task_ids.append(int(next_info["task_id"]))
        task_family_ids.append(int(next_info["task_family_id"]))
        split_ids.append(int(next_info["split_id"]))
        collection_mode_ids.append(COLLECTION_MODE_IDS["labelled" if labelled else "passive"])
        task_progress.append(float(next_info["task_progress"]))

        if terminated or truncated:
            break

    trajectory_length = len(actions)
    frames = _pad_sequence(frames, env_config.max_steps + 1)
    state_tokens = _pad_vector_sequence(state_tokens, env_config.max_steps + 1)
    state_masks = _pad_vector_sequence(state_masks, env_config.max_steps + 1)
    goal_vectors = _pad_vector_sequence(goal_vectors, env_config.max_steps + 1)
    task_ids = _pad_sequence([np.array([value], dtype=np.int64) for value in task_ids], env_config.max_steps + 1)
    task_family_ids = _pad_sequence(
        [np.array([value], dtype=np.int64) for value in task_family_ids], env_config.max_steps + 1
    )
    split_ids = _pad_sequence([np.array([value], dtype=np.int64) for value in split_ids], env_config.max_steps + 1)
    collection_mode_ids = _pad_sequence(
        [np.array([value], dtype=np.int64) for value in collection_mode_ids], env_config.max_steps + 1
    )
    task_progress = _pad_vector_sequence([np.array([value], dtype=np.float32) for value in task_progress], env_config.max_steps + 1)

    while len(actions) < env_config.max_steps:
        actions.append(-1)
        rewards.append(0.0)
        terminals.append(0.0)
        truncateds.append(0.0)
        action_mask.append(0.0)

    actions_array = np.asarray(actions, dtype=np.int64)
    rewards_array = np.asarray(rewards, dtype=np.float32)
    terminals_array = np.asarray(terminals, dtype=np.float32)
    truncateds_array = np.asarray(truncateds, dtype=np.float32)
    action_mask_array = np.asarray(action_mask, dtype=np.float32)

    frames_array = np.asarray(frames, dtype=np.uint8)
    state_tokens_array = np.asarray(state_tokens, dtype=np.float32)
    state_masks_array = np.asarray(state_masks, dtype=np.float32)
    goal_vectors_array = np.asarray(goal_vectors, dtype=np.float32)
    task_ids_array = np.asarray([value[0] for value in task_ids], dtype=np.int64)
    task_family_ids_array = np.asarray([value[0] for value in task_family_ids], dtype=np.int64)
    split_ids_array = np.asarray([value[0] for value in split_ids], dtype=np.int64)
    collection_mode_ids_array = np.asarray([value[0] for value in collection_mode_ids], dtype=np.int64)
    task_progress_array = np.asarray([value[0] for value in task_progress], dtype=np.float32)

    valid = trajectory_length
    flattened = {
        "images": frames_array[:valid],
        "next_images": frames_array[1 : valid + 1],
        "entity_features": state_tokens_array[:valid],
        "next_entity_features": state_tokens_array[1 : valid + 1],
        "entity_mask": state_masks_array[:valid],
        "goal_vectors": goal_vectors_array[:valid],
        "task_ids": task_ids_array[:valid],
        "task_family_ids": task_family_ids_array[:valid],
        "split_ids": split_ids_array[:valid],
        "collection_mode_ids": collection_mode_ids_array[:valid],
        "actions": actions_array[:valid],
        "labelled_mask": action_mask_array[:valid] if labelled else np.zeros(valid, dtype=np.float32),
        "rewards": rewards_array[:valid],
        "terminals": terminals_array[:valid],
        "truncateds": truncateds_array[:valid],
    }

    sequence = {
        "sequence_frames": frames_array,
        "sequence_state_tokens": state_tokens_array,
        "sequence_state_masks": state_masks_array,
        "sequence_goal_vectors": goal_vectors_array,
        "sequence_actions": actions_array,
        "sequence_rewards": rewards_array,
        "sequence_terminals": terminals_array,
        "sequence_truncateds": truncateds_array,
        "sequence_action_mask": action_mask_array,
        "sequence_trajectory_lengths": np.asarray(trajectory_length, dtype=np.int64),
        "sequence_task_ids": np.asarray(int(info["task_id"]), dtype=np.int64),
        "sequence_task_family_ids": np.asarray(int(info["task_family_id"]), dtype=np.int64),
        "sequence_split_ids": np.asarray(SPLIT_IDS.get(split_name, SPLIT_IDS["unspecified"]), dtype=np.int64),
        "sequence_collection_mode_ids": np.asarray(COLLECTION_MODE_IDS["labelled" if labelled else "passive"], dtype=np.int64),
        "sequence_task_progress": task_progress_array,
        "sequence_labelled_mask": action_mask_array if labelled else np.zeros_like(action_mask_array),
    }

    return {**sequence, **flattened}


def _append_arrays(storage: dict[str, list[np.ndarray]], episode: dict[str, np.ndarray]) -> None:
    for key, value in episode.items():
        storage.setdefault(key, []).append(value)


def _stack_sequences(storage: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    concatenated_keys = {
        "images",
        "next_images",
        "entity_features",
        "next_entity_features",
        "entity_mask",
        "goal_vectors",
        "task_ids",
        "task_family_ids",
        "split_ids",
        "collection_mode_ids",
        "actions",
        "labelled_mask",
        "rewards",
        "terminals",
        "truncateds",
    }
    arrays: dict[str, np.ndarray] = {}
    for key, value in storage.items():
        if key in concatenated_keys:
            arrays[key] = np.concatenate(value, axis=0)
        else:
            arrays[key] = np.asarray(value)
    return arrays


def save_dataset(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _collect_split(
    env_config: EnvConfig,
    num_transitions: int,
    policy_mix: tuple[str, ...],
    labelled: bool,
    seed: int,
    split_name: str,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    storage: dict[str, list[np.ndarray]] = defaultdict(list)
    collected_steps = 0
    episode_index = 0
    while collected_steps < num_transitions:
        episode = _collect_episode(
            env_config=env_config,
            policy_mix=policy_mix,
            labelled=labelled,
            seed=int(rng.integers(0, 1_000_000)),
            split_name=split_name,
        )
        _append_arrays(storage, episode)
        collected_steps += int(episode["sequence_trajectory_lengths"])
        episode_index += 1

    arrays = _stack_sequences(storage)
    arrays["num_episodes"] = np.asarray(episode_index, dtype=np.int64)
    arrays["transition_budget"] = np.asarray(num_transitions, dtype=np.int64)
    return arrays


def _split_name_from_filename(filename: str) -> str:
    if "train" in filename:
        return "train"
    if "val" in filename:
        return "val"
    if "test" in filename:
        return "test"
    return "unspecified"


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
    for filename, (budget, labelled, seed) in tqdm(split_specs.items(), desc="Generating splits"):
        split_name = _split_name_from_filename(filename)
        arrays = _collect_split(
            env_config=env_config,
            num_transitions=budget,
            policy_mix=dataset_config.policy_mix,
            labelled=labelled,
            seed=seed,
            split_name=split_name,
        )
        path = output_dir / filename
        save_dataset(path, arrays)
        outputs[filename] = path
    return outputs
