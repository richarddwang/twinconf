"""
Tests for: Variable and Expression Interpolation
Corresponds to: docs/interpolation.md
"""

import os

import pytest

from synconf.resolution.interpolator import (
    CircularInterpolationError,
    ExpressionError,
    Interpolator,
    UndefinedReferenceError,
)

# === Basic Variable Reference Tests ===


def test_interpolation_variable_references():
    """Tests for: Basic Variable References - type preservation, complete value refs."""
    interp = Interpolator()

    config = {
        "str_val": "adam",
        "int_val": 100,
        "float_val": 0.001,
        "list_val": [64, 128, 256],
        "dict_val": {"a": 1},
        "bool_val": True,
        "none_val": None,
        "ref_str": "~{str_val}",
        "ref_int": "~{int_val}",
        "ref_float": "~{float_val}",
        "ref_list": "~{list_val}",
        "ref_dict": "~{dict_val}",
        "ref_bool": "~{bool_val}",
        "ref_none": "~{none_val}",
    }
    result = interp.resolve_all(config)
    assert result["ref_str"] == "adam" and isinstance(result["ref_str"], str)
    assert result["ref_int"] == 100 and isinstance(result["ref_int"], int)
    assert result["ref_float"] == 0.001 and isinstance(result["ref_float"], float)
    assert result["ref_list"] == [64, 128, 256]
    assert result["ref_dict"] == {"a": 1}
    assert result["ref_bool"] is True
    assert result["ref_none"] is None


# === Nested Path Reference Tests ===


def test_interpolation_nested_paths():
    """Tests for: Nested Path References - dotted paths."""
    interp = Interpolator()

    config = {
        "optimizer": {"lr": 0.01},
        "model": {"encoder": {"hidden_size": 768}},
        "ref1": "~{optimizer.lr}",
        "ref2": "~{model.encoder.hidden_size}",
    }
    result = interp.resolve_all(config)
    assert result["ref1"] == 0.01 and result["ref2"] == 768


# === Partial/Embedded Interpolation Tests ===


def test_interpolation_partial():
    """Tests for: Partial/Embedded Interpolation - string construction, type re-parsing."""
    interp = Interpolator()

    config = {
        "name": "resnet",
        "version": "v2",
        "a": "es",
        "b": 4,
        "path": "model_~{name}_~{version}.pt",
        "bool_yes": "y~{a}",  # "yes" -> True
        "bool_no": "n~{b}",  # "n4" -> string (not "no")
        "int_val": "1~{b}",  # "14" -> 14
        "float_val": "3.1~{b}",  # "3.14" -> 3.14
    }
    result = interp.resolve_all(config)
    assert result["path"] == "model_resnet_v2.pt"
    assert result["bool_yes"] is True
    assert result["int_val"] == 14
    assert result["float_val"] == 3.14


# === Expression Interpolation Tests ===


def test_interpolation_expressions():
    """Tests for: Expression Interpolation - arithmetic, builtins, nested expressions."""
    interp = Interpolator()

    config = {
        "base": 10,
        "a": 5,
        "b": 3,
        "items": [1, 2, 3, 4, 5],
        "optimizer": {"lr": 0.001},
        "mult": "~{base * 2}",
        "add": "~{a + b}",
        "div": "~{a / b}",
        "nested_expr": "~{optimizer__lr * 10}",
        "len_fn": "~{len(items)}",
        "max_fn": "~{max(a, b)}",
        "min_fn": "~{min(a, b)}",
        "abs_fn": "~{abs(-42)}",
        "int_fn": "~{int(3.7)}",
        "comp": "~{a < b}",
    }
    result = interp.resolve_all(config)
    assert result["mult"] == 20 and result["add"] == 8
    assert result["div"] == pytest.approx(5 / 3)
    assert result["nested_expr"] == pytest.approx(0.01)
    assert result["len_fn"] == 5
    assert result["max_fn"] == 5 and result["min_fn"] == 3
    assert result["abs_fn"] == 42 and result["int_fn"] == 3
    assert result["comp"] is False


# === Environment Variables Tests ===


def test_interpolation_env_vars():
    """Tests for: Environment Variables - env: prefix."""
    interp = Interpolator()

    config = {"home": "~{env:HOME}", "path": "~{env:HOME}/data"}
    result = interp.resolve_all(config)
    assert result["home"] == os.environ["HOME"]
    assert result["path"] == f"{os.environ['HOME']}/data"


# === Chained Reference Tests ===


def test_interpolation_chained():
    """Tests for: Chained References - multi-level resolution."""
    interp = Interpolator()

    config = {"a": "~{b}", "b": "~{c}", "c": [1, 2, 3]}
    result = interp.resolve_all(config)
    assert result["a"] == [1, 2, 3] and result["b"] == [1, 2, 3]


# === List Interpolation Tests ===


def test_interpolation_lists():
    """Tests for: List Interpolation - values within lists."""
    interp = Interpolator()

    config = {"base": 10, "values": [1, "~{base}", 3]}
    result = interp.resolve_all(config)
    assert result["values"] == [1, 10, 3]


# === Edge Cases Tests ===


def test_interpolation_edge_cases():
    """Tests for: Edge Cases - no interpolation, empty config, tilde paths."""
    interp = Interpolator()

    # Test no interpolation (unchanged)
    config = {"plain": "string", "tilde": "~/home/user"}
    result = interp.resolve_all(config)
    assert result["plain"] == "string" and result["tilde"] == "~/home/user"

    # Test empty config
    assert interp.resolve_all({}) == {}


# === Error Handling Tests ===


def test_interpolation_circular_errors():
    """Tests for: Circular Reference Detection - self-ref, mutual, chain."""
    interp = Interpolator()

    # Test circular reference (a -> b -> a)
    with pytest.raises(CircularInterpolationError):
        interp.resolve_all({"a": "~{b}", "b": "~{a}"})

    # Test circular reference (a -> b -> c -> a)
    with pytest.raises(CircularInterpolationError):
        interp.resolve_all({"a": "~{b}", "b": "~{c}", "c": "~{a}"})

    # Test self-reference
    with pytest.raises(CircularInterpolationError):
        interp.resolve_all({"x": "~{x}"})


def test_interpolation_undefined_errors():
    """Tests for: Undefined Reference Errors - nonexistent key, nested, env var."""
    interp = Interpolator()

    # Test undefined reference
    with pytest.raises(UndefinedReferenceError):
        interp.resolve_all({"value": "~{nonexistent}"})

    # Test undefined nested reference
    with pytest.raises(UndefinedReferenceError):
        interp.resolve_all({"a": {"b": 1}, "value": "~{a.nonexistent}"})

    # Test undefined env var
    with pytest.raises(UndefinedReferenceError):
        interp.resolve_all({"value": "~{env:SYNCONF_NONEXISTENT_VAR_12345}"})


def test_interpolation_expression_errors():
    """Tests for: Expression Errors - dunder access, syntax error."""
    interp = Interpolator()

    # Test expression errors (dunder access)
    with pytest.raises(ExpressionError, match="not allowed"):
        interp.resolve_all({"value": "~{__import__('os')}"})

    # Test syntax error in expression
    with pytest.raises(ExpressionError, match="syntax error"):
        interp.resolve_all({"value": "~{1 + * 2}"})
