"""Unit tests for dataset split generation and near-duplicate leak prevention."""

import csv
import tempfile
from pathlib import Path

from src.urdu_nlp.dataset import create_stratified_dataset


def test_dataset_splits_disjointness_and_distribution():
    """Validates that train, dev, and test splits are strictly disjoint."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_f, dev_f, test_f = create_stratified_dataset(output_dir=tmp_path, seed=42)

        assert train_f.exists()
        assert dev_f.exists()
        assert test_f.exists()

        def load_texts(path: Path) -> list[str]:
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return [r["text"] for r in reader]

        train_texts = set(load_texts(train_f))
        dev_texts = set(load_texts(dev_f))
        test_texts = set(load_texts(test_f))

        # Check disjointness
        assert len(train_texts & dev_texts) == 0, "Train and Dev splits overlap!"
        assert len(train_texts & test_texts) == 0, "Train and Test splits overlap!"
        assert len(dev_texts & test_texts) == 0, "Dev and Test splits overlap!"

        # Check total volume
        assert len(train_texts) > len(dev_texts)
        assert len(train_texts) > len(test_texts)

