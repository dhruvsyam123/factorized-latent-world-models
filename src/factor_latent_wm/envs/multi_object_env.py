from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from factor_latent_wm.config.core import EnvConfig


GRID_BG = np.array([242, 239, 233], dtype=np.uint8)
COLORS = {
    "agent": np.array([52, 109, 219], dtype=np.uint8),
    "key": np.array([250, 191, 32], dtype=np.uint8),
    "door": np.array([105, 78, 56], dtype=np.uint8),
    "door_open": np.array([62, 166, 94], dtype=np.uint8),
    "block": np.array([245, 124, 0], dtype=np.uint8),
    "hazard": np.array([220, 53, 69], dtype=np.uint8),
    "target": np.array([53, 179, 124], dtype=np.uint8),
    "distractor": np.array([120, 120, 120], dtype=np.uint8),
}


class EntityType(IntEnum):
    AGENT = 0
    KEY = 1
    DOOR = 2
    BLOCK = 3
    HAZARD = 4
    TARGET = 5
    DISTRACTOR = 6


ACTION_DELTAS = {
    0: np.array([0, 0], dtype=np.int64),
    1: np.array([0, -1], dtype=np.int64),
    2: np.array([0, 1], dtype=np.int64),
    3: np.array([-1, 0], dtype=np.int64),
    4: np.array([1, 0], dtype=np.int64),
}

MAX_ENTITIES = 8
ENTITY_FEATURE_DIM = 14
TASK_FAMILIES = {
    "reach": "navigation",
    "key_door": "navigation_interaction",
    "push": "push_interaction",
}
TASK_FAMILY_IDS = {
    "navigation": 0,
    "navigation_interaction": 1,
    "push_interaction": 2,
}
SPLIT_IDS = {
    "train": 0,
    "val": 1,
    "test": 2,
    "unspecified": 3,
}


@dataclass
class Entity:
    kind: EntityType
    position: np.ndarray
    velocity: np.ndarray
    active: bool = True
    state: float = 0.0

    def copy(self) -> "Entity":
        return Entity(
            kind=self.kind,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            active=self.active,
            state=self.state,
        )


class MultiObjectEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 6}

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        self.grid_size = self.config.grid_size
        self.render_size = self.config.render_size
        self.cell_size = self.render_size // self.grid_size
        self.max_entities = 4 + self.config.max_hazards + self.config.max_distractors
        if self.max_entities > MAX_ENTITIES:
            raise ValueError(f"max_entities exceeds hard limit {MAX_ENTITIES}")

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.render_size, self.render_size, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(ACTION_DELTAS))
        self.np_random = np.random.default_rng(seed)
        self.step_count = 0
        self.current_task = "reach"
        self.current_split = "unspecified"
        self.agent = Entity(EntityType.AGENT, np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64))
        self.key = Entity(EntityType.KEY, np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), active=False)
        self.door = Entity(EntityType.DOOR, np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), active=False)
        self.block = Entity(EntityType.BLOCK, np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), active=False)
        self.target = Entity(EntityType.TARGET, np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64))
        self.hazards: list[Entity] = []
        self.distractors: list[Entity] = []
        self.has_key = False
        self.door_open = False

    def _sample_empty_cell(self, occupied: set[tuple[int, int]]) -> np.ndarray:
        while True:
            cell = self.np_random.integers(0, self.grid_size, size=2, endpoint=False)
            cell_t = (int(cell[0]), int(cell[1]))
            if cell_t not in occupied:
                occupied.add(cell_t)
                return cell.astype(np.int64)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = options or {}
        self.step_count = 0
        self.current_split = str(options.get("split", "unspecified"))
        sampled_task = options.get("task")
        sampled_family = options.get("task_family")
        if sampled_task is None:
            if sampled_family is not None:
                family_tasks = [task for task in self.config.task_types if TASK_FAMILIES.get(task) == sampled_family]
                if not family_tasks:
                    raise ValueError(f"Unknown or empty task family: {sampled_family}")
                sampled_task = str(self.np_random.choice(family_tasks))
            else:
                sampled_task = str(self.np_random.choice(self.config.task_types))
        self.current_task = str(sampled_task)
        occupied: set[tuple[int, int]] = set()
        self.has_key = False
        self.door_open = False

        self.agent = Entity(EntityType.AGENT, self._sample_empty_cell(occupied), np.zeros(2, dtype=np.int64))
        self.target = Entity(EntityType.TARGET, self._sample_empty_cell(occupied), np.zeros(2, dtype=np.int64))

        self.key.active = self.current_task == "key_door"
        self.door.active = self.current_task == "key_door"
        self.block.active = self.current_task == "push"

        if self.key.active:
            self.key.position = self._sample_empty_cell(occupied)
            self.key.velocity = np.zeros(2, dtype=np.int64)
            self.key.state = 0.0
            self.door.position = self._sample_empty_cell(occupied)
            self.door.velocity = np.zeros(2, dtype=np.int64)
            self.door.state = 0.0

        if self.block.active:
            self.block.position = self._sample_empty_cell(occupied)
            self.block.velocity = np.zeros(2, dtype=np.int64)
            self.block.state = 0.0

        self.hazards = []
        hazard_dirs = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
        for _ in range(self.config.max_hazards):
            velocity = hazard_dirs[int(self.np_random.integers(0, len(hazard_dirs)))].astype(np.int64)
            self.hazards.append(Entity(EntityType.HAZARD, self._sample_empty_cell(occupied), velocity))

        self.distractors = []
        for _ in range(self.config.max_distractors):
            self.distractors.append(
                Entity(EntityType.DISTRACTOR, self._sample_empty_cell(occupied), np.zeros(2, dtype=np.int64))
            )

        obs = self.render()
        return obs, self._build_info()

    def _inside(self, pos: np.ndarray) -> bool:
        return np.all((0 <= pos) & (pos < self.grid_size))

    def _blocked(self, pos: np.ndarray) -> bool:
        if np.array_equal(pos, self.door.position) and self.door.active and not self.door_open:
            return True
        return False

    def _move_or_push_block(self, delta: np.ndarray) -> None:
        candidate = self.agent.position + delta
        if not self._inside(candidate) or self._blocked(candidate):
            return
        if self.block.active and np.array_equal(candidate, self.block.position):
            block_candidate = self.block.position + delta
            if not self._inside(block_candidate) or self._blocked(block_candidate):
                return
            for entity in self.hazards + self.distractors:
                if entity.active and np.array_equal(block_candidate, entity.position):
                    return
            self.block.position = block_candidate
        self.agent.position = candidate

    def _move_hazard(self, hazard: Entity) -> None:
        candidate = hazard.position + hazard.velocity
        if not self._inside(candidate):
            hazard.velocity *= -1
            candidate = hazard.position + hazard.velocity
        if self._inside(candidate):
            hazard.position = candidate

    def _move_distractor(self, distractor: Entity) -> None:
        if self.np_random.random() < 0.5:
            return
        action = int(self.np_random.integers(0, len(ACTION_DELTAS)))
        candidate = distractor.position + ACTION_DELTAS[action]
        if self._inside(candidate):
            distractor.position = candidate

    def _update_interactions(self) -> None:
        if self.key.active and np.array_equal(self.agent.position, self.key.position):
            self.has_key = True
            self.key.active = False
            self.key.state = 1.0
        if self.door.active and self.has_key and np.array_equal(self.agent.position, self.door.position):
            self.door_open = True
            self.door.state = 1.0

    def _is_success(self) -> bool:
        if self.current_task == "reach":
            return np.array_equal(self.agent.position, self.target.position)
        if self.current_task == "key_door":
            return self.door_open and np.array_equal(self.agent.position, self.target.position)
        if self.current_task == "push":
            return np.array_equal(self.block.position, self.target.position)
        raise ValueError(f"Unknown task {self.current_task}")

    def _reward(self) -> float:
        if self.current_task == "reach":
            dist = np.linalg.norm(self.agent.position - self.target.position, ord=1)
            return 1.0 if self._is_success() else -0.05 * float(dist)
        if self.current_task == "key_door":
            key_dist = np.linalg.norm(self.agent.position - self.key.position, ord=1)
            door_dist = np.linalg.norm(self.agent.position - self.door.position, ord=1)
            target_dist = np.linalg.norm(self.agent.position - self.target.position, ord=1)
            reward = -0.02 * float(key_dist if not self.has_key else (door_dist if not self.door_open else target_dist))
            if self.has_key:
                reward += 0.2
            if self.door_open:
                reward += 0.4
            if self._is_success():
                reward += 1.0
            return reward
        if self.current_task == "push":
            block_dist = np.linalg.norm(self.block.position - self.target.position, ord=1)
            push_contact = np.linalg.norm(self.agent.position - self.block.position, ord=1)
            reward = -0.03 * float(block_dist) - 0.01 * float(push_contact)
            if self._is_success():
                reward += 1.0
            return reward
        raise ValueError(f"Unknown task {self.current_task}")

    def _task_id(self) -> int:
        return {"reach": 0, "key_door": 1, "push": 2}[self.current_task]

    def _task_family(self) -> str:
        return TASK_FAMILIES[self.current_task]

    def _task_family_id(self) -> int:
        return TASK_FAMILY_IDS[self._task_family()]

    def _split_id(self) -> int:
        return SPLIT_IDS.get(self.current_split, SPLIT_IDS["unspecified"])

    def _task_progress(self) -> float:
        if self.current_task == "reach":
            dist = np.linalg.norm(self.agent.position - self.target.position, ord=1)
            return float(-dist)
        if self.current_task == "key_door":
            if not self.has_key:
                return float(-np.linalg.norm(self.agent.position - self.key.position, ord=1))
            if not self.door_open:
                return float(-np.linalg.norm(self.agent.position - self.door.position, ord=1))
            return float(-np.linalg.norm(self.agent.position - self.target.position, ord=1))
        if self.current_task == "push":
            block_dist = np.linalg.norm(self.block.position - self.target.position, ord=1)
            return float(-block_dist)
        raise ValueError(f"Unknown task {self.current_task}")

    def _goal_vector(self) -> np.ndarray:
        goal = np.zeros(8, dtype=np.float32)
        goal[:2] = self.target.position / max(1, self.grid_size - 1)
        goal[2] = 1.0 if self.current_task == "key_door" else 0.0
        goal[3] = 1.0 if self.current_task == "push" else 0.0
        goal[4] = self.door.position[0] / max(1, self.grid_size - 1) if self.door.active else 0.0
        goal[5] = self.door.position[1] / max(1, self.grid_size - 1) if self.door.active else 0.0
        goal[6] = self.block.position[0] / max(1, self.grid_size - 1) if self.block.active else 0.0
        goal[7] = self.block.position[1] / max(1, self.grid_size - 1) if self.block.active else 0.0
        return goal

    def _entity_type_one_hot(self, kind: EntityType) -> np.ndarray:
        one_hot = np.zeros(7, dtype=np.float32)
        one_hot[int(kind)] = 1.0
        return one_hot

    def _entity_to_feature(self, entity: Entity) -> np.ndarray:
        feat = np.zeros(ENTITY_FEATURE_DIM, dtype=np.float32)
        feat[:7] = self._entity_type_one_hot(entity.kind)
        denom = max(1, self.grid_size - 1)
        feat[7] = entity.position[0] / denom
        feat[8] = entity.position[1] / denom
        feat[9] = float(entity.velocity[0])
        feat[10] = float(entity.velocity[1])
        feat[11] = 1.0 if entity.active else 0.0
        feat[12] = entity.state
        feat[13] = 1.0 if (entity.kind == EntityType.AGENT and self.has_key) else float(self.door_open)
        return feat

    def get_state_tokens(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_entity_tensor()

    def get_entity_tensor(self) -> tuple[np.ndarray, np.ndarray]:
        entities = [
            self.agent.copy(),
            self.key.copy(),
            self.door.copy(),
            self.block.copy(),
            *[h.copy() for h in self.hazards],
            *[d.copy() for d in self.distractors],
        ]
        feats = np.zeros((MAX_ENTITIES, ENTITY_FEATURE_DIM), dtype=np.float32)
        mask = np.zeros(MAX_ENTITIES, dtype=np.float32)
        for index, entity in enumerate(entities[:MAX_ENTITIES]):
            feats[index] = self._entity_to_feature(entity)
            mask[index] = 1.0
        return feats, mask

    def _build_info(self) -> dict[str, Any]:
        feats, mask = self.get_entity_tensor()
        return {
            "task": self.current_task,
            "task_id": self._task_id(),
            "task_family": self._task_family(),
            "task_family_id": self._task_family_id(),
            "split_name": self.current_split,
            "split_id": self._split_id(),
            "goal_vector": self._goal_vector(),
            "entity_features": feats,
            "entity_mask": mask,
            "state_tokens": feats,
            "state_mask": mask,
            "success": self._is_success(),
            "task_progress": self._task_progress(),
        }

    def step(self, action: int):
        self.step_count += 1
        delta = ACTION_DELTAS.get(int(action), ACTION_DELTAS[0])
        self._move_or_push_block(delta)
        self._update_interactions()

        for hazard in self.hazards:
            self._move_hazard(hazard)
        for distractor in self.distractors:
            self._move_distractor(distractor)

        terminated = any(np.array_equal(self.agent.position, hazard.position) for hazard in self.hazards)
        success = self._is_success()
        terminated = terminated or success
        truncated = self.step_count >= self.config.max_steps
        reward = self._reward()
        obs = self.render()
        info = self._build_info()
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        canvas = np.tile(GRID_BG, (self.render_size, self.render_size, 1))
        canvas[:: self.cell_size, :, :] = 210
        canvas[:, :: self.cell_size, :] = 210

        def draw(entity: Entity, color: np.ndarray) -> None:
            if not entity.active:
                return
            x0 = int(entity.position[0]) * self.cell_size
            y0 = int(entity.position[1]) * self.cell_size
            canvas[y0 : y0 + self.cell_size, x0 : x0 + self.cell_size] = color

        draw(self.target, COLORS["target"])
        draw(self.key, COLORS["key"])
        draw(self.door, COLORS["door_open"] if self.door_open else COLORS["door"])
        draw(self.block, COLORS["block"])
        for hazard in self.hazards:
            draw(hazard, COLORS["hazard"])
        for distractor in self.distractors:
            draw(distractor, COLORS["distractor"])
        draw(self.agent, COLORS["agent"])
        return canvas.astype(np.uint8)

    def snapshot(self) -> dict[str, Any]:
        frame = self.render()
        state_tokens, state_mask = self.get_state_tokens()
        info = self._build_info()
        return {
            "frame": frame,
            "state_tokens": state_tokens,
            "state_mask": state_mask,
            "task": self.current_task,
            "task_id": info["task_id"],
            "task_family": info["task_family"],
            "task_family_id": info["task_family_id"],
            "split_name": info["split_name"],
            "split_id": info["split_id"],
            "goal_vector": info["goal_vector"],
            "task_progress": info["task_progress"],
        }
