"""
Tests for: Validation
Corresponds to: docs/validation.md
"""

from pathlib import Path
from textwrap import dedent
from typing import Literal

import pytest
from conftest import Model, Trainer, create_optimizer

from synconf import SynConfParser


def _bad_seed(seed: int = "not_an_int"):
    """Function with incorrect default type for testing."""
    return seed


def test_validation_errors_with_sources(tmp_path: Path):
    """Tests for: All validation errors (unknown, missing, type mismatch) reported at once with source and owner."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        dedent("""\
        model:
          lr: 0.1
        trainer:
          optimizer: null
        optimizer_fn:
          lr: not_a_float
    """)
    )

    parser = SynConfParser()
    parser.add("model", Model)
    parser.add("trainer", Trainer)
    parser.add("optimizer_fn", create_optimizer)
    parser.add("bad_seed", _bad_seed)
    parser.add("num_trials", type_hint=int)

    with pytest.raises(ValueError) as exc_info:
        parser.parse(
            str(yaml_file),
            "--model.lr=not_a_float",
            "--model.unknown_param=value",
            "--nonexistent=foo",
        )

    expected = dedent("""\
        Configuration validation failed:
          Invalid type for 'model.lr': expected float, got str 'not_a_float' [source: CLI, owner: Model]
          Unexpected parameter: 'model.unknown_param' [source: CLI, owner: Model]
          Missing required parameter: 'trainer.max_epochs' [owner: Trainer]
          Invalid type for 'optimizer_fn.lr': expected float, got str 'not_a_float' [source: config.yaml, owner: create_optimizer]
          Invalid type for 'bad_seed.seed': expected int, got str 'not_an_int' [source: default value, owner: _bad_seed]
          Unexpected parameter: 'nonexistent' [source: CLI]
          Missing required parameter: 'num_trials'""")
    assert str(exc_info.value) == expected


def test_validation_on_assignment():
    """Tests for: Assigning an invalid type to a parsed config raises immediately."""
    parser = SynConfParser()
    parser.add("model", Model)
    config = parser.parse("--model.lr=0.1")

    # Invalid assignment raises TypeError
    with pytest.raises(TypeError) as exc_info:
        config.model.lr = "bad_string"

    expected = "Invalid type for 'lr': expected float, got str 'bad_string' [source: manual assignment, owner: Model]"
    assert str(exc_info.value) == expected

    # Valid assignment works
    config.model.lr = 0.05
    assert config.model.lr == 0.05


def test_validation_special_types():
    """Tests for: Literal, container types, Optional, Union, nested types validated correctly."""

    class SpecialModel:
        def __init__(
            self,
            mode: Literal["train", "eval", "test"],
            tags: list[str],
            metadata: dict[str, int],
            optional_lr: float | None = None,
            scores: list[int] | str = "none",
            nested: list[tuple[int, str]] | None = None,
        ):
            self.mode = mode
            self.tags = tags
            self.metadata = metadata
            self.optional_lr = optional_lr
            self.scores = scores
            self.nested = nested

    parser = SynConfParser()
    parser.add("valid", SpecialModel)
    parser.add("invalid", SpecialModel)

    # Valid and invalid coexist — errors are collected, valid group passes silently
    with pytest.raises(ValueError) as exc_info:
        parser.parse(
            "--valid.mode=train",
            "--valid.tags=[a,b]",
            "--valid.metadata={x: 1}",
            "--valid.optional_lr=null",
            "--valid.scores=[1,2,3]",
            "--invalid.mode=invalid_mode",
            "--invalid.tags=[1,2]",
            "--invalid.metadata={x: not_int}",
            "--invalid.nested=[[not_int, 123]]",
        )

    expected = dedent("""\
        Configuration validation failed:
          Invalid type for 'invalid.mode': expected Literal, got str 'invalid_mode' [source: CLI, owner: test_validation_special_types.<locals>.SpecialModel]
          Invalid type for 'invalid.tags': expected list, got list [1, 2] [source: CLI, owner: test_validation_special_types.<locals>.SpecialModel]
          Invalid type for 'invalid.metadata': expected dict, got dict {'x': 'not_int'} [source: CLI, owner: test_validation_special_types.<locals>.SpecialModel]
          Invalid type for 'invalid.nested': expected list[tuple[int, str]] | None, got list [['not_int', 123]] [source: CLI, owner: test_validation_special_types.<locals>.SpecialModel]""")
    assert str(exc_info.value) == expected
