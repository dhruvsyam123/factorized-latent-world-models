import torch

from factor_latent_wm.config.core import ModelConfig
from factor_latent_wm.models import ActionConditionedWorldModel, FactorisedLatentActionModel, MonolithicLatentWorldModel


def _make_batch() -> dict[str, torch.Tensor]:
    batch_size = 2
    max_entities = 8
    image_size = 84
    return {
        "image": torch.rand(batch_size, 3, image_size, image_size),
        "next_image": torch.rand(batch_size, 3, image_size, image_size),
        "entity_features": torch.rand(batch_size, max_entities, 14),
        "next_entity_features": torch.rand(batch_size, max_entities, 14),
        "entity_mask": torch.ones(batch_size, max_entities),
        "goal_vector": torch.rand(batch_size, 8),
        "task_id": torch.zeros(batch_size, dtype=torch.long),
        "action": torch.randint(0, 5, (batch_size,), dtype=torch.long),
        "labelled": torch.ones(batch_size),
    }


def test_factor_model_shapes():
    model = FactorisedLatentActionModel(ModelConfig())
    outputs = model(_make_batch())
    assert outputs["current_reconstruction"].shape == (2, 3, 84, 84)
    assert outputs["predicted_next_entities"].shape == (2, 8, 14)
    assert outputs["agent_action_logits"].shape == (2, 5)
    assert outputs["ctrl_posterior"].shape == (2, 16)
    assert outputs["exo_posterior"].shape == (2, 8, 16)


def test_slot_factor_model_shapes():
    model = FactorisedLatentActionModel(ModelConfig(encoder_type="pixel_slot"))
    outputs = model(_make_batch())
    assert outputs["current_reconstruction"].shape == (2, 3, 84, 84)
    assert outputs["predicted_next_entities"].shape == (2, 8, 14)
    assert outputs["factor_mask"].shape == (2, 8)


def test_monolithic_model_shapes():
    model = MonolithicLatentWorldModel(ModelConfig())
    outputs = model(_make_batch())
    assert outputs["next_reconstruction"].shape == (2, 3, 84, 84)


def test_action_conditioned_model_shapes():
    model = ActionConditionedWorldModel(ModelConfig())
    outputs = model(_make_batch())
    assert outputs["predicted_next_entities"].shape == (2, 8, 14)
