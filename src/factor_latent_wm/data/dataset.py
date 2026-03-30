from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _as_float_image(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float().permute(2, 0, 1) / 255.0


def _sequence_key_map(prefix: str = "sequence_") -> dict[str, str]:
    return {
        "frames": f"{prefix}frames",
        "state_tokens": f"{prefix}state_tokens",
        "state_masks": f"{prefix}state_masks",
        "goal_vectors": f"{prefix}goal_vectors",
        "actions": f"{prefix}actions",
        "rewards": f"{prefix}rewards",
        "terminals": f"{prefix}terminals",
        "truncateds": f"{prefix}truncateds",
        "action_mask": f"{prefix}action_mask",
        "trajectory_lengths": f"{prefix}trajectory_lengths",
        "task_ids": f"{prefix}task_ids",
        "task_family_ids": f"{prefix}task_family_ids",
        "split_ids": f"{prefix}split_ids",
        "collection_mode_ids": f"{prefix}collection_mode_ids",
        "task_progress": f"{prefix}task_progress",
        "labelled_mask": f"{prefix}labelled_mask",
    }


def _sequence_arrays_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    seq = _sequence_key_map()
    if seq["frames"] in data:
        return {key: data[value] for key, value in seq.items()}

    if "frames" in data:
        return {
            "frames": data["frames"],
            "state_tokens": data["state_tokens"],
            "state_masks": data["state_masks"],
            "goal_vectors": data["goal_vectors"],
            "actions": data["actions"],
            "rewards": data["rewards"],
            "terminals": data["terminals"],
            "truncateds": data["truncateds"],
            "action_mask": data["action_mask"],
            "trajectory_lengths": data["trajectory_lengths"],
            "task_ids": data["task_ids"],
            "task_family_ids": data["task_family_ids"],
            "split_ids": data["split_ids"],
            "collection_mode_ids": data["collection_mode_ids"],
        }

    if "images" not in data or "next_images" not in data:
        raise ValueError("Dataset file does not contain sequence or transition arrays")

    images = data["images"]
    next_images = data["next_images"]
    entity_features = data["entity_features"]
    next_entity_features = data["next_entity_features"]
    entity_mask = data["entity_mask"]
    goal_vectors = data["goal_vectors"]
    task_ids = data["task_ids"]
    task_family_ids = data["task_family_ids"] if "task_family_ids" in data else np.zeros_like(task_ids)
    split_ids = data["split_ids"] if "split_ids" in data else np.zeros_like(task_ids)
    collection_mode_ids = (
        data["collection_mode_ids"] if "collection_mode_ids" in data else np.zeros_like(task_ids)
    )
    actions = data["actions"]
    labelled_mask = data["labelled_mask"] if "labelled_mask" in data else (actions != -1).astype(np.float32)
    rewards = data["rewards"] if "rewards" in data else np.zeros(actions.shape[0], dtype=np.float32)
    terminals = data["terminals"] if "terminals" in data else np.zeros(actions.shape[0], dtype=np.float32)
    truncateds = data["truncateds"] if "truncateds" in data else np.zeros(actions.shape[0], dtype=np.float32)

    return {
        "frames": np.stack([images, next_images], axis=1),
        "state_tokens": np.stack([entity_features, next_entity_features], axis=1),
        "state_masks": np.stack([entity_mask, entity_mask], axis=1),
        "goal_vectors": np.stack([goal_vectors, goal_vectors], axis=1),
        "actions": actions[:, None],
        "rewards": rewards[:, None],
        "terminals": terminals[:, None],
        "truncateds": truncateds[:, None],
        "action_mask": np.ones(actions.shape[0], dtype=np.float32)[:, None],
        "trajectory_lengths": np.ones(actions.shape[0], dtype=np.int64),
        "task_ids": task_ids,
        "task_family_ids": task_family_ids,
        "split_ids": split_ids,
        "collection_mode_ids": collection_mode_ids,
        "task_progress": np.zeros(actions.shape[0], dtype=np.float32)[:, None],
        "labelled_mask": labelled_mask[:, None] if labelled_mask.ndim == 1 else labelled_mask,
    }


def _flatten_sequence_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    frames = arrays["frames"]
    state_tokens = arrays["state_tokens"]
    state_masks = arrays["state_masks"]
    goal_vectors = arrays["goal_vectors"]
    actions = arrays["actions"]
    rewards = arrays["rewards"]
    terminals = arrays["terminals"]
    truncateds = arrays["truncateds"]
    action_mask = arrays["action_mask"]
    lengths = arrays["trajectory_lengths"]
    task_ids = arrays["task_ids"]
    task_family_ids = arrays["task_family_ids"]
    split_ids = arrays["split_ids"]
    collection_mode_ids = arrays["collection_mode_ids"]
    task_progress = arrays["task_progress"]
    labelled_mask = arrays["labelled_mask"]

    flat: dict[str, list[np.ndarray]] = {
        "images": [],
        "next_images": [],
        "entity_features": [],
        "next_entity_features": [],
        "entity_mask": [],
        "goal_vectors": [],
        "task_ids": [],
        "task_family_ids": [],
        "split_ids": [],
        "collection_mode_ids": [],
        "actions": [],
        "labelled_mask": [],
        "rewards": [],
        "terminals": [],
        "truncateds": [],
        "trajectory_index": [],
        "step_index": [],
        "task_progress": [],
    }

    for trajectory_index, length in enumerate(lengths):
        valid_steps = int(length)
        if valid_steps <= 0:
            continue
        flat["images"].append(frames[trajectory_index, :valid_steps])
        flat["next_images"].append(frames[trajectory_index, 1 : valid_steps + 1])
        flat["entity_features"].append(state_tokens[trajectory_index, :valid_steps])
        flat["next_entity_features"].append(state_tokens[trajectory_index, 1 : valid_steps + 1])
        flat["entity_mask"].append(state_masks[trajectory_index, :valid_steps])
        flat["goal_vectors"].append(goal_vectors[trajectory_index, :valid_steps])
        flat["task_ids"].append(np.full(valid_steps, task_ids[trajectory_index], dtype=np.int64))
        flat["task_family_ids"].append(np.full(valid_steps, task_family_ids[trajectory_index], dtype=np.int64))
        flat["split_ids"].append(np.full(valid_steps, split_ids[trajectory_index], dtype=np.int64))
        flat["collection_mode_ids"].append(
            np.full(valid_steps, collection_mode_ids[trajectory_index], dtype=np.int64)
        )
        flat["actions"].append(actions[trajectory_index, :valid_steps])
        flat["labelled_mask"].append(labelled_mask[trajectory_index, :valid_steps])
        flat["rewards"].append(rewards[trajectory_index, :valid_steps])
        flat["terminals"].append(terminals[trajectory_index, :valid_steps])
        flat["truncateds"].append(truncateds[trajectory_index, :valid_steps])
        flat["trajectory_index"].append(np.full(valid_steps, trajectory_index, dtype=np.int64))
        flat["step_index"].append(np.arange(valid_steps, dtype=np.int64))
        flat["task_progress"].append(task_progress[trajectory_index, :valid_steps])

    return {key: np.concatenate(value, axis=0) for key, value in flat.items()}


def _stack_item(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value)


class SequenceDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            arrays = _sequence_arrays_from_npz(data)
        self.frames = arrays["frames"]
        self.state_tokens = arrays["state_tokens"]
        self.state_masks = arrays["state_masks"]
        self.goal_vectors = arrays["goal_vectors"]
        self.actions = arrays["actions"]
        self.rewards = arrays["rewards"]
        self.terminals = arrays["terminals"]
        self.truncateds = arrays["truncateds"]
        self.action_mask = arrays["action_mask"]
        self.trajectory_lengths = arrays["trajectory_lengths"]
        self.task_ids = arrays["task_ids"]
        self.task_family_ids = arrays["task_family_ids"]
        self.split_ids = arrays["split_ids"]
        self.collection_mode_ids = arrays["collection_mode_ids"]
        self.task_progress = arrays["task_progress"]
        self.labelled_mask = arrays["labelled_mask"]

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        length = int(self.trajectory_lengths[index])
        return {
            "frames": _stack_item(self.frames[index]).float().permute(0, 3, 1, 2) / 255.0,
            "state_tokens": _stack_item(self.state_tokens[index]).float(),
            "state_masks": _stack_item(self.state_masks[index]).float(),
            "goal_vectors": _stack_item(self.goal_vectors[index]).float(),
            "actions": _stack_item(self.actions[index]).long(),
            "rewards": _stack_item(self.rewards[index]).float(),
            "terminals": _stack_item(self.terminals[index]).float(),
            "truncateds": _stack_item(self.truncateds[index]).float(),
            "action_mask": _stack_item(self.action_mask[index]).float(),
            "trajectory_length": torch.tensor(length, dtype=torch.long),
            "task_id": torch.tensor(self.task_ids[index], dtype=torch.long),
            "task_family_id": torch.tensor(self.task_family_ids[index], dtype=torch.long),
            "split_id": torch.tensor(self.split_ids[index], dtype=torch.long),
            "collection_mode_id": torch.tensor(self.collection_mode_ids[index], dtype=torch.long),
            "task_progress": _stack_item(self.task_progress[index]).float(),
            "labelled_mask": _stack_item(self.labelled_mask[index]).float(),
        }


class TransitionDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            arrays = _sequence_arrays_from_npz(data)

        flattened = _flatten_sequence_arrays(arrays)

        self.images = flattened["images"]
        self.next_images = flattened["next_images"]
        self.entity_features = flattened["entity_features"]
        self.next_entity_features = flattened["next_entity_features"]
        self.entity_mask = flattened["entity_mask"]
        self.goal_vectors = flattened["goal_vectors"]
        self.task_ids = flattened["task_ids"]
        self.task_family_ids = flattened["task_family_ids"]
        self.split_ids = flattened["split_ids"]
        self.collection_mode_ids = flattened["collection_mode_ids"]
        self.task_progress = flattened["task_progress"]
        self.actions = flattened["actions"]
        self.labelled_mask = flattened["labelled_mask"]
        self.rewards = flattened["rewards"]
        self.terminals = flattened["terminals"]
        self.truncateds = flattened["truncateds"]
        self.trajectory_index = flattened["trajectory_index"]
        self.step_index = flattened["step_index"]

        self.state_tokens = self.entity_features
        self.next_state_tokens = self.next_entity_features
        self.state_masks = self.entity_mask

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": _as_float_image(self.images[index]),
            "next_image": _as_float_image(self.next_images[index]),
            "entity_features": torch.from_numpy(self.entity_features[index]).float(),
            "next_entity_features": torch.from_numpy(self.next_entity_features[index]).float(),
            "entity_mask": torch.from_numpy(self.entity_mask[index]).float(),
            "state_tokens": torch.from_numpy(self.state_tokens[index]).float(),
            "next_state_tokens": torch.from_numpy(self.next_state_tokens[index]).float(),
            "state_masks": torch.from_numpy(self.state_masks[index]).float(),
            "goal_vector": torch.from_numpy(self.goal_vectors[index]).float(),
            "task_id": torch.tensor(self.task_ids[index], dtype=torch.long),
            "task_family_id": torch.tensor(self.task_family_ids[index], dtype=torch.long),
            "split_id": torch.tensor(self.split_ids[index], dtype=torch.long),
            "collection_mode_id": torch.tensor(self.collection_mode_ids[index], dtype=torch.long),
            "task_progress": torch.tensor(self.task_progress[index], dtype=torch.float32),
            "action": torch.tensor(self.actions[index], dtype=torch.long),
            "labelled": torch.tensor(self.labelled_mask[index], dtype=torch.float32),
            "reward": torch.tensor(self.rewards[index], dtype=torch.float32),
            "terminal": torch.tensor(self.terminals[index], dtype=torch.float32),
            "truncated": torch.tensor(self.truncateds[index], dtype=torch.float32),
            "trajectory_index": torch.tensor(self.trajectory_index[index], dtype=torch.long),
            "step_index": torch.tensor(self.step_index[index], dtype=torch.long),
        }
