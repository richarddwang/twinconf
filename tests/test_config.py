"""
Tests for: Config Class
Corresponds to: docs/core-concepts.md#initialization
"""

import pytest

from synconf import Config
from synconf.resolution.resolver import TYPE_KEY

# === Test Fixtures ===


class Optimizer:
    """Base optimizer class."""

    def __init__(self, lr: float, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2


class Trainer:
    """Training controller."""

    def __init__(self, max_epochs: int, optimizer: Optimizer | None = None):
        self.max_epochs = max_epochs
        self.optimizer = optimizer


class Callback:
    """Base callback."""

    def __init__(self, name: str):
        self.name = name


class LoggingCallback(Callback):
    """Logging callback."""

    def __init__(self, level: str = "INFO", **kwargs):
        super().__init__(**kwargs)
        self.level = level


class CallbackTrainer:
    """Trainer with callbacks."""

    def __init__(self, callbacks: list | None = None):
        self.callbacks = callbacks or []


# === Access Pattern Tests ===


def test_config_access_patterns():
    """Tests for: Config access - dict, attribute, nested, contains, iter, len, get, keys/values/items, to_dict, setters."""
    # Dict and attribute access
    config = Config({"name": "test", "value": 42})
    assert config["name"] == "test" and config.name == "test"
    assert config["value"] == 42 and config.value == 42

    # Nested access (auto-wraps as Config)
    config = Config({"optimizer": {"lr": 0.01, "beta1": 0.9}})
    assert isinstance(config["optimizer"], Config)
    assert config["optimizer"]["lr"] == 0.01 and config.optimizer.lr == 0.01

    # __contains__ and __iter__
    config = Config({"a": 1, "b": 2, "c": 3})
    assert "a" in config and "missing" not in config
    assert set(config) == {"a", "b", "c"}

    # len
    assert len(config) == 3

    # get with default
    config = Config({"name": "test"})
    assert config.get("name") == "test"
    assert config.get("missing", "default") == "default"

    # keys, values, items
    config = Config({"a": 1, "b": 2})
    assert "a" in config.keys()
    assert 1 in config.values()
    assert ("a", 1) in config.items()

    # to_dict (recursive)
    config = Config({"name": "test", "nested": {"value": 42}})
    result = config.to_dict()
    assert isinstance(result, dict) and not isinstance(result, Config)
    assert result["nested"]["value"] == 42

    # setitem and setattr
    config = Config({"name": "test"})
    config["name"] = "updated"
    assert config["name"] == "updated"
    config.name = "changed"
    assert config.name == "changed"


# === Initialization Tests ===


def test_config_init_simple():
    """Tests for: Config.init - simple instantiation."""
    config = Config({TYPE_KEY: Optimizer, "lr": 0.01, "weight_decay": 0.001})
    optimizer = config.init()
    assert isinstance(optimizer, Optimizer)
    assert optimizer.lr == 0.01 and optimizer.weight_decay == 0.001


def test_config_instantiate_alias():
    """Tests for: Config.instantiate() alias."""
    config = Config({TYPE_KEY: Optimizer, "lr": 0.01})
    optimizer = config.instantiate()
    assert isinstance(optimizer, Optimizer)


def test_config_init_with_overwrites():
    """Tests for: Config.init with overwrites."""
    config = Config({TYPE_KEY: Adam, "lr": 0.01, "beta1": 0.9})
    optimizer = config.init(lr=0.001)
    assert optimizer.lr == 0.001 and optimizer.beta1 == 0.9


def test_config_init_nested_type():
    """Tests for: Nested TYPE (recursive init)."""
    config = Config({TYPE_KEY: Trainer, "max_epochs": 100, "optimizer": {TYPE_KEY: Adam, "lr": 0.01}})
    trainer = config.init()
    assert isinstance(trainer, Trainer)
    assert isinstance(trainer.optimizer, Adam)
    assert trainer.optimizer.lr == 0.01


def test_config_init_missing_type():
    """Tests for: Missing TYPE error."""
    config = Config({"lr": 0.01})
    with pytest.raises(ValueError, match="no TYPE"):
        config.init()


def test_config_init_list_with_types():
    """Tests for: List containing TYPE dicts."""
    config = Config(
        {
            TYPE_KEY: CallbackTrainer,
            "callbacks": [
                {TYPE_KEY: LoggingCallback, "name": "logger", "level": "DEBUG"},
                {TYPE_KEY: Callback, "name": "base"},
            ],
        }
    )
    trainer = config.init()
    assert isinstance(trainer.callbacks, list)
    assert len(trainer.callbacks) == 2
    assert isinstance(trainer.callbacks[0], LoggingCallback)
    assert trainer.callbacks[0].level == "DEBUG"
    assert isinstance(trainer.callbacks[1], Callback)
