"""
Tests for: Variable and Expression Interpolation
Corresponds to: docs/interpolation.md
"""

import os
from pathlib import Path
from textwrap import dedent

import pytest

from synconf import SynConfParser


def test_interpolation(tmp_path: Path):
    """Tests for: All interpolation patterns in one setup."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        dedent("""\
        base_val: hello
        int_val: 100
        optimizer:
          lr: 0.001
        a: 5
        b: 3

        var_ref: ~{base_val}
        var_nested: ~{optimizer.lr}
        expr_ref: ~{a * 2}
        expr_nested: ~{optimizer.lr * 10}
        partial_str: model_~{base_val}_v2.pt
        partial_float: 3.1~{b}
        list_interp: [1, "~{int_val}", 3]
        partial_two: ~{base_val}_~{int_val}
        env_ref: ~{env:HOME}
    """)
    )

    parser = SynConfParser()
    config = parser.parse(str(yaml_file))

    # Variable interpolation
    assert config.var_ref == "hello"
    # Variable interpolation (nested)
    assert config.var_nested == 0.001
    # Expression interpolation
    assert config.expr_ref == 10
    # Expression interpolation (nested)
    assert config.expr_nested == pytest.approx(0.01)
    # Partial interpolation in string
    assert config.partial_str == "model_hello_v2.pt"
    # Partial interpolation in float
    assert config.partial_float == 3.13
    # Interpolation in list
    assert config.list_interp == [1, 100, 3]
    # Partial interpolation (two interpolations)
    assert config.partial_two == "hello_100"
    # Environment variable interpolation
    assert config.env_ref == os.environ["HOME"]


def test_interpolation_errors(tmp_path: Path):
    """Tests for: All interpolation errors (circular, undefined, unsafe expression) collected at once."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        dedent("""\
        circular_a: ~{circular_b}
        circular_b: ~{circular_a}
        undefined: ~{nonexistent}
        dangerous: ~{__import__('os')}
    """)
    )

    parser = SynConfParser()
    with pytest.raises(ValueError) as exc_info:
        parser.parse(str(yaml_file))

    expected = dedent("""\
        Configuration validation failed:
          Circular reference detected: circular_b -> circular_a -> circular_b
          Circular reference detected: circular_a -> circular_b -> circular_a
          Undefined reference: 'nonexistent'
          Invalid expression '~{__import__('os')}': access to '__import__' not allowed""")
    assert str(exc_info.value) == expected
