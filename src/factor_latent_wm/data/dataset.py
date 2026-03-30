from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with np.load(self.path) as data:
            self.images = data["images"]
            self.next_images = data["next_images"]
            self.entity_features = data["entity_features"]
            self.next_entity_features = data["next_entity_features"]
            self.entity_mask = data["entity_mask"]
            self.goal_vectors = data["goal_vectors"]
            self.task_ids = data["task_ids"]
            self.actions = data["actions"]
            self.labelled_mask = data["labelled_mask"]

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(self.images[index]).float().permute(2, 0, 1) / 255.0
        next_image = torch.from_numpy(self.next_images[index]).float().permute(2, 0, 1) / 255.0
        return {
            "image": image,
            "next_image": next_image,
            "entity_features": torch.from_numpy(self.entity_features[index]).float(),
            "next_entity_features": torch.from_numpy(self.next_entity_features[index]).float(),
            "entity_mask": torch.from_numpy(self.entity_mask[index]).float(),
            "goal_vector": torch.from_numpy(self.goal_vectors[index]).float(),
            "task_id": torch.tensor(self.task_ids[index], dtype=torch.long),
            "action": torch.tensor(self.actions[index], dtype=torch.long),
            "labelled": torch.tensor(self.labelled_mask[index], dtype=torch.float32),
        }
