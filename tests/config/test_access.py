"""
Tests for: Config Access Patterns
Corresponds to: docs/core-concepts.md#initialization
"""

from synconf import Config


def test_config_access_patterns():
    """Tests for: Config access - dict, attribute, nested, contains, iter, len, get, keys/values/items, to_dict, setters."""
    config = Config({"name": "test", "value": 42, "optimizer": {"lr": 0.01, "beta1": 0.9}})

    # Dict and attribute access
    assert config["name"] == "test" and config.name == "test"
    assert config["value"] == 42 and config.value == 42

    # Nested access (auto-wraps as Config)
    assert isinstance(config["optimizer"], Config)
    assert config["optimizer"]["lr"] == 0.01 and config.optimizer.lr == 0.01

    # Dot-notation nested get and set
    assert config["optimizer.lr"] == 0.01
    config["optimizer.lr"] = 0.02
    assert config.optimizer.lr == 0.02

    # __contains__ and __iter__
    assert "name" in config and "missing" not in config
    assert set(config) == {"name", "value", "optimizer"}

    # len
    assert len(config) == 3

    # get with default
    assert config.get("name") == "test"
    assert config.get("missing", "default") == "default"

    # keys, values, items
    assert "name" in config.keys()
    assert 42 in config.values()
    assert ("name", "test") in config.items()

    # to_dict (recursive)
    result = config.to_dict()
    assert isinstance(result, dict) and not isinstance(result, Config)
    assert result["optimizer"]["lr"] == 0.02

    # setitem and setattr
    config["name"] = "updated"
    assert config["name"] == "updated"
    config.name = "changed"
    assert config.name == "changed"
