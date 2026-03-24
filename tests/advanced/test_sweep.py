"""
Tests for: Parameter Sweep
Corresponds to: docs/advanced.md#parameter-sweep
"""

from textwrap import dedent

import pytest
from conftest import Model

from synconf import SynConfParser


def test_sweep_parse_args():
    """Tests for: Sweep - returns list with cartesian product."""
    parser = SynConfParser()
    parser.add("model", Model)

    configs = parser.parse_args(
        [
            "--model.lr=0.1",
            "sweep",
            "--model.lr=[0.01,0.001]",
            "--model.peft_config=[null,{r: 2}]",
        ]
    )
    assert isinstance(configs, list) and len(configs) == 4
    assert configs[0].model.lr == 0.01 and configs[0].model.peft_config is None
    assert configs[1].model.lr == 0.01 and configs[1].model.to_dict()["peft_config"] == {"r": 2}
    assert configs[2].model.lr == 0.001 and configs[2].model.peft_config is None
    assert configs[3].model.lr == 0.001 and configs[3].model.to_dict()["peft_config"] == {"r": 2}


def test_sweep_validation_error():
    """Tests for: Sweep - validation error in sweep config."""
    parser = SynConfParser()
    parser.add("model", Model)

    with pytest.raises(ValueError) as exc_info:
        parser.parse_args(["--model.lr=0.1", "sweep", "--model.lr=[0.01,not_a_float]"])

    expected = dedent("""\
        Configuration validation failed:
          Invalid type for 'model.lr': expected float, got str 'not_a_float' [source: CLI, owner: Model]""")
    assert str(exc_info.value) == expected
