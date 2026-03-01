"""
Tests for: Validation
Corresponds to: docs/validation.md
"""

from typing import Optional

import pytest

from synconf.introspection.spec import CallableSpec, ParamSpec
from synconf.validation.validator import Validator

# === Test Fixtures ===


class BaseOptimizer:
    """Base optimizer."""

    def __init__(self, lr: float, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay


# === Type Hints Validation Tests ===


def test_validator_basic_types():
    """Tests for: Type Hints Validation - int, float, str, bool."""
    validator = Validator()

    # Test basic type validation
    assert validator.validate(42, int) is True
    assert validator.validate(3.14, float) is True
    assert validator.validate("hello", str) is True
    assert validator.validate(True, bool) is True
    assert validator.validate("42", int) is False
    assert validator.validate(42, str) is False


def test_validator_optional():
    """Tests for: Optional type validation."""
    validator = Validator()

    assert validator.validate(42, Optional[int]) is True
    assert validator.validate(None, Optional[int]) is True
    assert validator.validate("42", Optional[int]) is False


def test_validator_collections():
    """Tests for: List and dict validation."""
    validator = Validator()

    # Test list validation
    assert validator.validate([1, 2, 3], list[int]) is True
    assert validator.validate([1, 2, 3], list) is True
    assert validator.validate("not a list", list[int]) is False

    # Test dict validation
    assert validator.validate({"a": 1}, dict[str, int]) is True


def test_validator_no_hint():
    """Tests for: No type hint always passes."""
    validator = Validator()
    assert validator.validate("anything", None) is True


# === Config Validation Tests ===


def test_validator_config_valid():
    """Tests for: validate_config - valid configuration."""
    validator = Validator()
    spec = CallableSpec(name="test", callable_obj=BaseOptimizer)
    spec.params["lr"] = ParamSpec(name="lr", type_hint=float)
    spec.params["weight_decay"] = ParamSpec(name="weight_decay", type_hint=float, default=0.0)

    # Valid config
    errors = validator.validate_config({"lr": 0.01}, spec)
    assert errors == []


def test_validator_config_invalid_type():
    """Tests for: validate_config - invalid type error."""
    validator = Validator()
    spec = CallableSpec(name="test", callable_obj=BaseOptimizer)
    spec.params["lr"] = ParamSpec(name="lr", type_hint=float)

    # Invalid type
    errors = validator.validate_config({"lr": "not_float"}, spec)
    assert len(errors) == 1 and "Invalid type" in errors[0]


def test_validator_config_missing_required():
    """Tests for: validate_config - missing required parameter."""
    validator = Validator()
    spec = CallableSpec(name="test", callable_obj=BaseOptimizer)
    spec.params["lr"] = ParamSpec(name="lr", type_hint=float)

    # Missing required
    errors = validator.validate_config({}, spec)
    assert len(errors) == 1 and "Missing required" in errors[0]


def test_validator_validate_or_raise():
    """Tests for: validate_or_raise - raises ValueError."""
    validator = Validator()
    spec = CallableSpec(name="test", callable_obj=BaseOptimizer)
    spec.params["lr"] = ParamSpec(name="lr", type_hint=float)

    with pytest.raises(ValueError, match="validation failed"):
        validator.validate_or_raise({}, spec)


# === Strict Validation Tests ===


def test_validator_strict_mode():
    """Tests for: Strict Validation - unknown parameter detection."""
    validator = Validator()
    spec = CallableSpec(name="test", callable_obj=BaseOptimizer)
    spec.params["lr"] = ParamSpec(name="lr", type_hint=float)
    spec.params["weight_decay"] = ParamSpec(name="weight_decay", type_hint=float, default=0.0)

    # Test strict mode - unknown parameter
    errors = validator.validate_config({"lr": 0.01, "unknown": 123}, spec, strict=True)
    assert len(errors) == 1 and "Unknown parameter" in errors[0]

    # Test non-strict mode - unknown parameter allowed
    errors = validator.validate_config({"lr": 0.01, "unknown": 123}, spec, strict=False)
    assert errors == []
