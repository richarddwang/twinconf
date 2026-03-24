"""
Tests for: Source Loading and Merging
Corresponds to: docs/quickstart.md#mixing-yaml-and-cli, docs/core-concepts.md
"""

from pathlib import Path
from textwrap import dedent

import pytest
from conftest import Adam

from synconf import SynConfParser


def test_yaml_loading(tmp_path: Path):
    """Tests for: YAML Loading - basic types, nested, lists, multiple files."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        dedent("""\
        optimizer:
          lr: 0.01
          beta1: 0.9
          beta2: 0.999
    """)
    )

    parser = SynConfParser()
    parser.add("optimizer", Adam)
    config = parser.parse(str(yaml_file))

    assert config.optimizer.lr == 0.01
    assert config.optimizer.beta1 == 0.9
    assert config.optimizer.beta2 == 0.999


def test_cli_overrides_yaml(tmp_path: Path):
    """Tests for: CLI Loading - type inference, nested paths, CLI overrides YAML."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        dedent("""\
        optimizer:
          lr: 0.1
          beta1: 0.9
    """)
    )

    parser = SynConfParser()
    parser.add("optimizer", Adam)

    # CLI overrides YAML values; type inference (float, bool)
    config = parser.parse(str(yaml_file), "--optimizer.lr=0.001")
    assert config.optimizer.lr == 0.001
    assert config.optimizer.beta1 == 0.9  # Preserved from YAML


def test_multiple_source_merging(tmp_path: Path):
    """Tests for: Merging - multiple YAML, CLI, order-based priority."""
    base = tmp_path / "base.yaml"
    base.write_text(
        dedent("""\
        optimizer:
          lr: 0.1
          beta1: 0.9
          beta2: 0.999
    """)
    )
    override = tmp_path / "override.yaml"
    override.write_text(
        dedent("""\
        optimizer:
          beta1: 0.8
    """)
    )

    parser = SynConfParser()
    parser.add("optimizer", Adam)

    # Sources applied left-to-right: base.yaml -> CLI -> override.yaml
    config = parser.parse(str(base), "--optimizer.lr=0.01", str(override))
    assert config.optimizer.lr == 0.01  # From CLI (overrides base)
    assert config.optimizer.beta1 == 0.8  # From override.yaml (overrides CLI)
    assert config.optimizer.beta2 == 0.999  # Preserved from base
