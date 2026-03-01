"""
Tests for: Resolution - TYPE Resolution and Compatibility
Corresponds to: docs/validation.md#type-compatibility
"""

import pytest

from synconf.resolution.resolver import Resolver

# === Test Fixtures ===


class BaseOptimizer:
    """Base optimizer."""

    def __init__(self, lr: float):
        self.lr = lr


class Adam(BaseOptimizer):
    """Adam optimizer."""

    def __init__(self, beta1: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = beta1


class AdamW(Adam):
    """AdamW optimizer."""

    def __init__(self, weight_decay: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.weight_decay = weight_decay


class ConvModel:
    """Model class (not an optimizer)."""

    def __init__(self, layers: int):
        self.layers = layers


def create_optimizer(lr: float = 0.001) -> BaseOptimizer:
    """Factory function returning BaseOptimizer."""
    return Adam(lr=lr)


def create_other() -> str:
    """Factory returning wrong type."""
    return "not an optimizer"


# === Resolver Tests ===


def test_resolver_registration():
    """Tests for: Resolver - register, resolve by name, resolve callable, resolve import path."""
    resolver = Resolver()

    # Test register and resolve by name
    resolver.register("BaseOptimizer", BaseOptimizer)
    assert resolver.resolve("BaseOptimizer") is BaseOptimizer

    # Test resolve callable directly
    assert resolver.resolve(Adam) is Adam

    # Test resolve import path
    from collections import OrderedDict

    assert resolver.resolve("collections.OrderedDict") is OrderedDict

    # Test invalid name raises ValueError
    with pytest.raises(ValueError, match="Cannot resolve"):
        resolver.resolve("NonExistentClass")

    # Test register_many
    resolver.register_many({"Adam": Adam, "AdamW": AdamW})
    assert resolver.resolve("Adam") is Adam and resolver.resolve("AdamW") is AdamW


def test_resolver_compatibility():
    """Tests for: TYPE Compatibility - subclass, same class, invalid, function return type."""
    resolver = Resolver()

    # Test compatibility checking
    assert resolver.check_compatibility(Adam, BaseOptimizer) is True  # Subclass
    assert resolver.check_compatibility(BaseOptimizer, BaseOptimizer) is True  # Same class

    # Test incompatible class
    with pytest.raises(TypeError, match="is not a subclass"):
        resolver.check_compatibility(ConvModel, BaseOptimizer)

    # Test function return type compatibility
    assert resolver.check_compatibility(create_optimizer, BaseOptimizer) is True
    with pytest.raises(TypeError, match="not compatible"):
        resolver.check_compatibility(create_other, BaseOptimizer)


def test_resolver_resolve_in_config():
    """Tests for: resolve_in_config - basic, nested, expected_types, incompatible."""
    resolver = Resolver()
    resolver.register("Adam", Adam)
    resolver.register("BaseOptimizer", BaseOptimizer)
    resolver.register("ConvModel", ConvModel)

    # Test basic TYPE resolution
    config = {"model": {"TYPE": "Adam", "lr": 0.01}}
    resolved = resolver.resolve_in_config(config)
    assert resolved["model"]["TYPE"] is Adam and resolved["model"]["lr"] == 0.01

    # Test nested TYPE resolution
    config = {"trainer": {"model": {"TYPE": "Adam", "beta1": 0.9}, "epochs": 100}}
    resolved = resolver.resolve_in_config(config)
    assert resolved["trainer"]["model"]["TYPE"] is Adam

    # Test with expected_types
    config = {"model": {"TYPE": "Adam"}}
    resolved = resolver.resolve_in_config(config, {"model": BaseOptimizer})
    assert resolved["model"]["TYPE"] is Adam

    # Test incompatible type during resolution
    config = {"model": {"TYPE": "ConvModel"}}
    with pytest.raises(TypeError, match="is not a subclass"):
        resolver.resolve_in_config(config, {"model": BaseOptimizer})
