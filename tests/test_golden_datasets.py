"""Tests for golden_datasets."""

import tempfile
from pathlib import Path

import pytest
from multiagent_eval.golden_datasets.manager import GoldenDatasetManager
from multiagent_eval.golden_datasets.schema import GoldenExample, HumanLabel


def test_create_dataset() -> None:
    """Test dataset creation."""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = GoldenDatasetManager(base_path=tmp)
        path = mgr.create_dataset("test_ds", "desc")
        assert Path(path).exists()
        data = mgr._load_dataset("test_ds")
        assert data["name"] == "test_ds"
        assert data["description"] == "desc"


def test_add_example() -> None:
    """Test adding example."""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = GoldenDatasetManager(base_path=tmp)
        mgr.create_dataset("ds")
        eid = mgr.add_example("ds", {"q": "x"}, final_expected_output={"a": "y"})
        assert eid
        data = mgr._load_dataset("ds")
        assert len(data["examples"]) == 1
        assert data["examples"][0]["pipeline_input"] == {"q": "x"}


def test_add_human_label() -> None:
    """Test adding human label."""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = GoldenDatasetManager(base_path=tmp)
        mgr.create_dataset("ds")
        eid = mgr.add_example("ds", {"q": "x"})
        mgr.add_human_label("ds", eid, "agent1", 0.8, "rater1")
        data = mgr._load_dataset("ds")
        assert len(data["examples"][0]["human_labels"]) == 1


def test_export_import() -> None:
    """Test export and import."""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = GoldenDatasetManager(base_path=tmp)
        mgr.create_dataset("ds")
        mgr.add_example("ds", {"x": 1})
        export_path = Path(tmp) / "export.json"
        mgr.export_json("ds", str(export_path))
        assert export_path.exists()
        name = mgr.import_json(str(export_path), "imported")
        assert name == "imported"
        data = mgr._load_dataset("imported")
        assert len(data["examples"]) == 1
