"""
Tests for: TYPE Resolution and Source Parsing
Corresponds to: docs/core-concepts.md#the-type-key, docs/validation.md#type-compatibility
"""

from pathlib import Path
from textwrap import dedent

import pytest
from conftest import AdamW, Optimizer, Scheduler, StepLR, Trainer

from synconf import SynConfParser


def test_type_resolution_from_sources(tmp_path: Path):
    """Tests for: TYPE resolution - default from add(), TYPE in YAML, TYPE via CLI."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        dedent("""\
        trainer:
          max_epochs: 100
        scheduler:
          TYPE: conftest.StepLR
          step_size: 20
        optimizer:
          TYPE: conftest.Adam
          lr: 0.01
    """)
    )

    parser = SynConfParser()
    parser.add("trainer", Trainer)
    parser.add("scheduler", Scheduler)
    parser.add("optimizer", Optimizer)
    config = parser.parse(str(yaml_file), "--optimizer.TYPE=conftest.AdamW")

    # Default TYPE from add() when no TYPE in YAML
    assert config.trainer["TYPE"] is Trainer
    # YAML TYPE overrides default
    assert config.scheduler["TYPE"] is StepLR
    # CLI TYPE overrides YAML TYPE
    assert config.optimizer["TYPE"] is AdamW


def test_type_compatibility(tmp_path: Path):
    """Tests for: TYPE compatibility - valid subclass accepted, incompatible type reported as error."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        dedent("""\
        optimizer:
          TYPE: conftest.Adam
          lr: 0.01
        scheduler:
          TYPE: conftest.Adam
          step_size: 10
    """)
    )

    parser = SynConfParser()
    parser.add("optimizer", Optimizer)
    parser.add("scheduler", Scheduler)

    with pytest.raises(ValueError) as exc_info:
        parser.parse(str(yaml_file))

    expected = dedent("""\
        Configuration validation failed:
          TYPE Adam is not a subclass of expected type Scheduler [source: config.yaml]""")
    assert str(exc_info.value) == expected
