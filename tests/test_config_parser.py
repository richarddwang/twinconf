"""
Tests for: Config and SynConfParser Integration
Corresponds to: src/synconf/config.py, src/synconf/parser.py
"""

from pathlib import Path

import pytest

from synconf import Config, SynConfParser
from synconf.resolution.resolver import TYPE_KEY

# --- Test Fixtures ---


class Optimizer:
    """Base optimizer class."""

    def __init__(self, lr: float, weight_decay: float = 0.0):
        """Initialize optimizer.

        Args:
            lr: Learning rate.
            weight_decay: Weight decay factor.
        """
        self.lr = lr
        self.weight_decay = weight_decay


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, **kwargs):
        """Initialize Adam.

        Args:
            beta1: First moment decay.
            beta2: Second moment decay.
        """
        super().__init__(**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2


class Trainer:
    """Training controller."""

    def __init__(self, max_epochs: int, optimizer: Optimizer):
        """Initialize trainer.

        Args:
            max_epochs: Maximum training epochs.
            optimizer: Optimizer instance.
        """
        self.max_epochs = max_epochs
        self.optimizer = optimizer


# --- Config Tests ---


class TestConfigAccess:
    """Tests for Config access patterns."""

    def test_dict_access(self):
        """Tests for: Dict-style access config['key']."""
        config = Config({"name": "test", "value": 42})

        assert config["name"] == "test"
        assert config["value"] == 42

    def test_attribute_access(self):
        """Tests for: Attribute-style access config.key."""
        config = Config({"name": "test", "value": 42})

        assert config.name == "test"
        assert config.value == 42

    def test_nested_access(self):
        """Tests for: Nested config access."""
        config = Config({"optimizer": {"lr": 0.01, "beta1": 0.9}})

        # Should wrap nested dicts as Config
        assert isinstance(config["optimizer"], Config)
        assert config["optimizer"]["lr"] == 0.01
        assert config.optimizer.lr == 0.01

    def test_contains(self):
        """Tests for: __contains__ (in operator)."""
        config = Config({"name": "test"})

        assert "name" in config
        assert "missing" not in config

    def test_iter(self):
        """Tests for: Iteration over keys."""
        config = Config({"a": 1, "b": 2})

        keys = list(config)
        assert "a" in keys
        assert "b" in keys

    def test_len(self):
        """Tests for: len() of config."""
        config = Config({"a": 1, "b": 2, "c": 3})
        assert len(config) == 3

    def test_get_with_default(self):
        """Tests for: get() method with default."""
        config = Config({"name": "test"})

        assert config.get("name") == "test"
        assert config.get("missing", "default") == "default"

    def test_keys_values_items(self):
        """Tests for: keys(), values(), items() methods."""
        config = Config({"a": 1, "b": 2})

        assert "a" in config.keys()
        assert 1 in config.values()
        assert ("a", 1) in config.items()

    def test_to_dict(self):
        """Tests for: Converting config to plain dict."""
        config = Config({"name": "test", "nested": {"value": 42}})

        result = config.to_dict()

        assert isinstance(result, dict)
        assert not isinstance(result, Config)
        assert result["name"] == "test"
        assert result["nested"]["value"] == 42

    def test_setitem(self):
        """Tests for: Setting values via dict-style."""
        config = Config({"name": "test"})
        config["name"] = "updated"

        assert config["name"] == "updated"

    def test_setattr(self):
        """Tests for: Setting values via attribute-style."""
        config = Config({"name": "test"})
        config.name = "updated"

        assert config.name == "updated"


class TestConfigInstantiate:
    """Tests for Config.instantiate() method."""

    def test_instantiate_simple(self):
        """Tests for: Basic instantiation with TYPE."""
        config = Config(
            {
                TYPE_KEY: Optimizer,
                "lr": 0.01,
                "weight_decay": 0.001,
            }
        )

        optimizer = config.instantiate()

        assert isinstance(optimizer, Optimizer)
        assert optimizer.lr == 0.01
        assert optimizer.weight_decay == 0.001

    def test_instantiate_with_overwrites(self):
        """Tests for: Instantiation with overwrite values."""
        config = Config(
            {
                TYPE_KEY: Adam,
                "lr": 0.01,
                "beta1": 0.9,
            }
        )

        # Override lr
        optimizer = config.instantiate(lr=0.001)

        assert optimizer.lr == 0.001
        assert optimizer.beta1 == 0.9

    def test_instantiate_nested(self):
        """Tests for: Instantiation with nested TYPE configs."""
        config = Config(
            {
                TYPE_KEY: Trainer,
                "max_epochs": 100,
                "optimizer": {
                    TYPE_KEY: Adam,
                    "lr": 0.01,
                },
            }
        )

        trainer = config.instantiate()

        assert isinstance(trainer, Trainer)
        assert trainer.max_epochs == 100
        assert isinstance(trainer.optimizer, Adam)
        assert trainer.optimizer.lr == 0.01

    def test_instantiate_no_type(self):
        """Tests for: ValueError when TYPE is missing."""
        config = Config({"lr": 0.01})

        with pytest.raises(ValueError, match="no TYPE"):
            config.instantiate()


# --- SynConfParser Tests ---


class TestSynConfParserAdd:
    """Tests for SynConfParser.add() method."""

    def test_add_class(self):
        """Tests for: Adding a class to parser."""
        parser = SynConfParser()
        parser.add("optimizer", Adam)

        spec = parser.get_spec("optimizer")
        assert spec is not None
        assert spec.name == "Adam"
        assert "beta1" in spec.params

    def test_add_multiple_groups(self):
        """Tests for: Adding multiple groups."""
        parser = SynConfParser()
        parser.add("optimizer", Adam)
        parser.add("trainer", Trainer)

        assert parser.get_spec("optimizer") is not None
        assert parser.get_spec("trainer") is not None


class TestSynConfParserParse:
    """Tests for SynConfParser.parse() method."""

    def test_parse_yaml(self, tmp_path: Path):
        """Tests for: Parsing YAML file."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("""
optimizer:
  lr: 0.01
  beta1: 0.9
""")

        parser = SynConfParser()
        parser.add("optimizer", Adam)

        config = parser.parse(str(yaml_file))

        assert config.optimizer.lr == 0.01
        assert config.optimizer.beta1 == 0.9

    def test_parse_cli_args(self, tmp_path: Path):
        """Tests for: Parsing CLI arguments."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("""
optimizer:
  lr: 0.01
""")

        parser = SynConfParser()
        parser.add("optimizer", Adam)

        # CLI arg overrides YAML
        config = parser.parse(str(yaml_file), "--optimizer.lr=0.001")

        assert config.optimizer.lr == 0.001

    def test_parse_mixed_sources(self, tmp_path: Path):
        """Tests for: Mixed YAML and CLI in order."""
        base_yaml = tmp_path / "base.yaml"
        base_yaml.write_text("""
optimizer:
  lr: 0.1
  beta1: 0.9
""")

        override_yaml = tmp_path / "override.yaml"
        override_yaml.write_text("""
optimizer:
  beta1: 0.8
""")

        parser = SynConfParser()
        parser.add("optimizer", Adam)

        # Order: base.yaml -> --lr=0.01 -> override.yaml
        config = parser.parse(
            str(base_yaml),
            "--optimizer.lr=0.01",
            str(override_yaml),
        )

        # lr from CLI (middle), beta1 from override.yaml (last)
        assert config.optimizer.lr == 0.01
        assert config.optimizer.beta1 == 0.8

    def test_parse_type_in_yaml(self, tmp_path: Path):
        """Tests for: TYPE specification in YAML."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("""
optimizer:
  TYPE: Adam
  lr: 0.01
""")

        parser = SynConfParser()
        parser.add("optimizer", Optimizer)  # Base type
        # Register Adam so it can be resolved by name
        parser._resolver.register("Adam", Adam)

        config = parser.parse(str(yaml_file))

        # TYPE should be resolved to Adam class
        assert config.optimizer[TYPE_KEY] is Adam

    def test_parse_type_in_cli(self, tmp_path: Path):
        """Tests for: TYPE specification via CLI."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("""
optimizer:
  lr: 0.01
""")

        parser = SynConfParser()
        parser.add("optimizer", Optimizer)
        # Register Adam so it can be resolved by name
        parser._resolver.register("Adam", Adam)

        config = parser.parse(str(yaml_file), "--optimizer.TYPE=Adam")

        assert config.optimizer[TYPE_KEY] is Adam

    def test_parse_default_type(self, tmp_path: Path):
        """Tests for: Default TYPE when not specified."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("""
optimizer:
  lr: 0.01
""")

        parser = SynConfParser()
        parser.add("optimizer", Adam)

        config = parser.parse(str(yaml_file))

        # Should default to Adam (the registered type)
        assert config.optimizer[TYPE_KEY] is Adam


class TestSynConfParserHelp:
    """Tests for SynConfParser.help() method."""

    def test_help_basic(self):
        """Tests for: Basic help generation."""
        parser = SynConfParser()
        parser.add("optimizer", Adam)

        help_text = parser.help()

        # Should contain the group in CLI format with .TYPE suffix
        assert "--optimizer.TYPE=Adam" in help_text

        # Should contain parameters in CLI format
        assert "--optimizer.beta1" in help_text
        assert "--optimizer.beta2" in help_text

        # Should show type in angle brackets
        assert "<float>" in help_text

        # Should show default values in metadata
        assert "default=0.9" in help_text
        assert "default=0.999" in help_text

        # Should show descriptions
        assert "First moment decay" in help_text
        assert "Second moment decay" in help_text

    def test_help_shows_kwargs_sources(self):
        """Tests for: Help shows from=Source for kwargs params."""
        parser = SynConfParser()
        parser.add("optimizer", Adam)

        help_text = parser.help()

        # lr comes from Optimizer via kwargs, should show source
        assert "--optimizer.lr" in help_text
        assert "from=Optimizer" in help_text

        # weight_decay also from Optimizer
        assert "--optimizer.weight_decay" in help_text

        # beta1 and beta2 should show from=Adam
        assert "from=Adam" in help_text

    def test_help_shows_missing_required(self):
        """Tests for: Help shows ??? for missing required params."""
        parser = SynConfParser()
        parser.add("optimizer", Adam)

        help_text = parser.help()

        # lr is required (no default), should show ???
        assert "--optimizer.lr=???" in help_text
        assert "required" in help_text

    def test_help_with_config_values(self):
        """Tests for: Help shows current config values."""
        parser = SynConfParser()
        parser.add("optimizer", Adam)

        # Provide config data
        config_data = {"optimizer": {"TYPE": Adam, "lr": 0.001, "beta1": 0.95, "weight_decay": 0.01}}

        help_text = parser.help(config_data)

        # Should show current values
        assert "--optimizer.lr=0.001" in help_text
        assert "--optimizer.beta1=0.95" in help_text
        assert "--optimizer.weight_decay=0.01" in help_text

        # beta2 not in config, should show default
        assert "--optimizer.beta2=0.999" in help_text

    def test_help_multiple_groups(self):
        """Tests for: Help with multiple groups."""
        parser = SynConfParser()
        parser.add("optimizer", Adam)
        parser.add("trainer", Trainer)

        help_text = parser.help()

        # Should contain both groups with .TYPE suffix
        assert "--optimizer.TYPE=Adam" in help_text
        assert "--trainer.TYPE=Trainer" in help_text

        # Should contain params from both
        assert "--optimizer.beta1" in help_text
        assert "--trainer.max_epochs" in help_text


class TestSynConfParserEndToEnd:
    """End-to-end integration tests."""

    def test_full_workflow(self, tmp_path: Path):
        """Tests for: Complete parse -> access -> instantiate workflow."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("""
trainer:
  max_epochs: 100
  optimizer:
    TYPE: Adam
    lr: 0.01
    beta1: 0.9
""")

        parser = SynConfParser()
        parser.add("trainer", Trainer)
        # Register Adam so it can be resolved by name in nested config
        parser._resolver.register("Adam", Adam)

        config = parser.parse(str(yaml_file))

        # Dict-style access
        assert config["trainer"]["max_epochs"] == 100

        # Attribute-style access
        assert config.trainer.max_epochs == 100

        # Nested config access
        assert config.trainer.optimizer.lr == 0.01

        # Instantiate sub-config
        optimizer = config.trainer.optimizer.instantiate()
        assert isinstance(optimizer, Adam)
        assert optimizer.lr == 0.01

        # Instantiate with override
        optimizer2 = config.trainer.optimizer.instantiate(lr=0.001)
        assert optimizer2.lr == 0.001

    def test_type_compatibility_check(self, tmp_path: Path):
        """Tests for: TYPE compatibility is checked."""

        class WrongClass:
            pass

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("""
optimizer:
  TYPE: WrongClass
  lr: 0.01
""")

        parser = SynConfParser()
        parser.add("optimizer", Optimizer)

        # Register the wrong class so it can be resolved
        parser._resolver.register("WrongClass", WrongClass)

        with pytest.raises(TypeError, match="is not a subclass"):
            parser.parse(str(yaml_file))
