"""
Tests for: Loading - YAML, CLI, and Merging
Corresponds to: Core configuration loading and merging functionality
"""

import tempfile
from pathlib import Path

import pytest

from synconf.loading.cli_loader import CliLoader, is_cli_arg
from synconf.loading.merger import REMOVE_KEY, Merger, deep_merge
from synconf.loading.yaml_loader import YamlLoader

# === YAML Loading Tests ===


def test_yaml_loading():
    """Tests for: YAML Loading - basic, nested, lists, multiple files, invalid YAML."""
    loader = YamlLoader()

    # Test basic and nested loading
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_file = Path(tmpdir) / "config.yaml"
        yaml_file.write_text("name: test\nvalue: 42\nnested:\n  key: value")

        result = loader.load(str(yaml_file))
        assert result["name"] == "test" and result["value"] == 42
        assert result["nested"]["key"] == "value"

        # Test list loading
        yaml_file.write_text("items:\n  - one\n  - two\n  - three")
        result = loader.load(str(yaml_file))
        assert result["items"] == ["one", "two", "three"]

        # Test multiple files (load each separately)
        file1, file2 = Path(tmpdir) / "a.yaml", Path(tmpdir) / "b.yaml"
        file1.write_text("x: 1")
        file2.write_text("y: 2")
        result1 = loader.load(str(file1))
        result2 = loader.load(str(file2))
        assert result1 == {"x": 1} and result2 == {"y": 2}

        # Test invalid YAML raises
        yaml_file.write_text("invalid:\nyaml: [")
        with pytest.raises(Exception):
            loader.load(str(yaml_file))


# === CLI Loading Tests ===


def test_cli_loading():
    """Tests for: CLI Loading - basic, nested, type inference, TYPE key, errors."""
    loader = CliLoader()

    # Test basic parsing
    assert loader.parse("--lr=0.01") == {"lr": 0.01}
    assert loader.parse("--epochs=100") == {"epochs": 100}

    # Test nested paths
    assert loader.parse("--optimizer.lr=0.001") == {"optimizer": {"lr": 0.001}}
    assert loader.parse("--trainer.optimizer.scheduler.gamma=0.1") == {
        "trainer": {"optimizer": {"scheduler": {"gamma": 0.1}}}
    }

    # Test type inference (int, float, bool, string, list, dict)
    assert loader.parse("--epochs=100")["epochs"] == 100
    assert loader.parse("--lr=0.01")["lr"] == 0.01
    assert loader.parse("--enabled=true")["enabled"] is True
    assert loader.parse("--disabled=false")["disabled"] is False
    assert loader.parse("--name=test")["name"] == "test"
    assert loader.parse("--layers=[1, 2, 3]")["layers"] == [1, 2, 3]
    assert loader.parse("--config={a: 1}")["config"] == {"a": 1}

    # Test TYPE key parsing
    assert loader.parse("--optimizer.TYPE=Adam") == {"optimizer": {"TYPE": "Adam"}}

    # Test errors
    with pytest.raises(ValueError, match="must start with"):
        loader.parse("lr=0.01")
    with pytest.raises(ValueError, match="must contain"):
        loader.parse("--lr")

    # Test parse_many
    results = loader.parse_many(["--lr=0.01", "--epochs=100"])
    assert len(results) == 2

    # Test is_cli_arg helper
    assert is_cli_arg("--lr=0.01") and is_cli_arg("--optimizer.TYPE=Adam")
    assert not is_cli_arg("config.yaml")


# === Merging Tests ===


def test_merging():
    """Tests for: Merging - simple, nested, deep, override scalar, REMOVE, preserve original."""
    merger = Merger()

    # Test simple merge
    assert merger.merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merger.merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}

    # Test nested merge (preserves non-overridden keys)
    base = {"optimizer": {"lr": 0.01, "beta1": 0.9}}
    override = {"optimizer": {"lr": 0.001}}
    assert merger.merge(base, override) == {"optimizer": {"lr": 0.001, "beta1": 0.9}}

    # Test deeply nested merge
    base = {"a": {"b": {"c": 1, "d": 2}}}
    override = {"a": {"b": {"c": 10}}}
    assert merger.merge(base, override) == {"a": {"b": {"c": 10, "d": 2}}}

    # Test override dict with scalar
    assert merger.merge({"config": {"nested": "value"}}, {"config": "scalar"}) == {"config": "scalar"}

    # Test merge_all
    assert merger.merge_all([]) == {}
    assert merger.merge_all([{"a": 1}, {"b": 2}, {"c": 3}]) == {"a": 1, "b": 2, "c": 3}
    assert merger.merge_all([{"a": 1}, {"a": 2}, {"a": 3}]) == {"a": 3}

    # Test preserves original (doesn't mutate)
    base, override = {"a": 1}, {"b": 2}
    result = merger.merge(base, override)
    assert base == {"a": 1} and override == {"b": 2}

    # Test deep_merge convenience function
    assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    # Test REMOVE feature
    base = {"model": {"lr": 0.1, "peft_config": {"r": 2}}}
    override = {"model": {"peft_config": REMOVE_KEY}}
    result = merger.merge(base, override)
    assert "lr" in result["model"] and "peft_config" not in result["model"]

    # Test REMOVE on nonexistent key (no effect)
    assert merger.merge({"a": 1}, {"nonexistent": REMOVE_KEY}) == {"a": 1}

    # Test REMOVE nested
    base = {"a": {"b": {"c": 1, "d": 2}}}
    result = merger.merge(base, {"a": {"b": {"c": REMOVE_KEY}}})
    assert result == {"a": {"b": {"d": 2}}}
