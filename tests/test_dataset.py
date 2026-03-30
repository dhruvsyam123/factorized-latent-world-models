from pathlib import Path

from factor_latent_wm.config.core import DatasetConfig, EnvConfig
from factor_latent_wm.data.dataset import SequenceDataset, TransitionDataset
from factor_latent_wm.data.generate import generate_default_splits


def test_dataset_generation_and_loading(tmp_path: Path):
    dataset_config = DatasetConfig(
        passive_train_transitions=32,
        passive_val_transitions=8,
        labelled_train_transitions=16,
        labelled_val_transitions=4,
    )
    outputs = generate_default_splits(tmp_path, EnvConfig(max_hazards=1, max_distractors=1), dataset_config)
    sequence_dataset = SequenceDataset(outputs["passive_train.npz"])
    dataset = TransitionDataset(outputs["passive_train.npz"])
    seq_sample = sequence_dataset[0]
    sample = dataset[0]
    assert len(dataset) >= 32
    assert seq_sample["frames"].shape[0] == 41
    assert sample["image"].shape[0] == 3
    assert sample["entity_features"].shape[0] >= 4
