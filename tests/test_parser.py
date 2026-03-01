"""
Tests for: SynConfParser Integration
Corresponds to: docs/core-concepts.md
"""

import os
import tempfile
from pathlib import Path

import pytest

from synconf import Config, SynConfParser
from synconf.resolution.resolver import TYPE_KEY


# === Test Fixtures ===


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


class Scheduler:
    """Learning rate scheduler."""

    def __init__(self, step_size: int = 10):
        """Initialize scheduler.

        Args:
            step_size: Step size for decay.
        """
        self.step_size = step_size


class StepLR(Scheduler):
    """Step learning rate scheduler."""

    def __init__(self, gamma: float = 0.1, **kwargs):
        """Initialize StepLR.

        Args:
            gamma: Decay factor.
        """
        super().__init__(**kwargs)
        self.gamma = gamma


class Trainer:
    """Training controller."""

    def __init__(self, max_epochs: int, optimizer: Optimizer | None = None):
        """Initialize trainer.

        Args:
            max_epochs: Maximum training epochs.
            optimizer: Optimizer instance.
        """
        self.max_epochs = max_epochs
        self.optimizer = optimizer


# === Parser Add and Spec Tests ===


def test_parser_add_and_spec():
    """Tests for: SynConfParser.add() and get_spec()."""
    parser = SynConfParser()
    parser.add("optimizer", Adam)

    spec = parser.get_spec("optimizer")
    assert spec is not None
    assert spec.name == "Adam"
    assert "beta1" in spec.params

    # Multiple groups
    parser.add("trainer", Trainer)
    assert parser.get_spec("optimizer") is not None
    assert parser.get_spec("trainer") is not None


# === Parser Parse Tests ===


def test_parser_parse_yaml_and_cli(tmp_path: Path):
    """Tests for: SynConfParser.parse() - YAML file, CLI args, mixed sources, TYPE in YAML/CLI."""
    # YAML only
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("optimizer:\n  lr: 0.01\n  beta1: 0.9")

    parser = SynConfParser()
    parser.add("optimizer", Adam)
    config = parser.parse(str(yaml_file))
    assert config.optimizer.lr == 0.01 and config.optimizer.beta1 == 0.9

    # CLI overrides YAML
    config = parser.parse(str(yaml_file), "--optimizer.lr=0.001")
    assert config.optimizer.lr == 0.001

    # Mixed sources (order: base.yaml -> --cli -> override.yaml)
    base = tmp_path / "base.yaml"
    base.write_text("optimizer:\n  lr: 0.1\n  beta1: 0.9")
    override = tmp_path / "override.yaml"
    override.write_text("optimizer:\n  beta1: 0.8")

    config = parser.parse(str(base), "--optimizer.lr=0.01", str(override))
    assert config.optimizer.lr == 0.01  # From CLI
    assert config.optimizer.beta1 == 0.8  # From override

    # TYPE in YAML
    yaml_file.write_text("optimizer:\n  TYPE: Adam\n  lr: 0.01")
    parser = SynConfParser()
    parser.add("optimizer", Optimizer)
    parser._resolver.register("Adam", Adam)
    config = parser.parse(str(yaml_file))
    assert config.optimizer[TYPE_KEY] is Adam

    # TYPE via CLI
    yaml_file.write_text("optimizer:\n  lr: 0.01")
    config = parser.parse(str(yaml_file), "--optimizer.TYPE=Adam")
    assert config.optimizer[TYPE_KEY] is Adam

    # Default TYPE when not specified
    parser = SynConfParser()
    parser.add("optimizer", Adam)
    yaml_file.write_text("optimizer:\n  lr: 0.01")
    config = parser.parse(str(yaml_file))
    assert config.optimizer[TYPE_KEY] is Adam


# === Parser Help Tests ===


def test_parser_help():
    """Tests for: SynConfParser.help() - basic, kwargs sources, missing required, config values, multiple groups."""
    parser = SynConfParser()
    parser.add("optimizer", Adam)

    help_text = parser.help()

    # Contains group with .TYPE suffix
    assert "--optimizer.TYPE=Adam" in help_text

    # Contains parameters in CLI format
    assert "--optimizer.beta1" in help_text and "--optimizer.beta2" in help_text

    # Shows type in angle brackets
    assert "<float>" in help_text

    # Shows default values
    assert "default=0.9" in help_text

    # Shows descriptions
    assert "First moment decay" in help_text

    # Shows from=Source for kwargs params
    assert "--optimizer.lr" in help_text and "from=Optimizer" in help_text

    # Shows ??? for missing required
    assert "--optimizer.lr=???" in help_text and "required" in help_text

    # With config values
    config_data = {"optimizer": {"TYPE": Adam, "lr": 0.001, "beta1": 0.95}}
    help_text = parser.help(config_data)
    assert "--optimizer.lr=0.001" in help_text
    assert "--optimizer.beta1=0.95" in help_text

    # Multiple groups
    parser.add("trainer", Trainer)
    help_text = parser.help()
    assert "--optimizer.TYPE=Adam" in help_text
    assert "--trainer.TYPE=Trainer" in help_text


# === End-to-End Tests ===


def test_parser_end_to_end(tmp_path: Path):
    """Tests for: Complete parse -> access -> init workflow with nested TYPE."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        "trainer:\n  max_epochs: 100\n  optimizer:\n    TYPE: Adam\n    lr: 0.01\n    beta1: 0.9"
    )

    parser = SynConfParser()
    parser.add("trainer", Trainer)
    parser._resolver.register("Adam", Adam)

    config = parser.parse(str(yaml_file))

    # Dict and attribute access
    assert config["trainer"]["max_epochs"] == 100
    assert config.trainer.max_epochs == 100
    assert config.trainer.optimizer.lr == 0.01

    # Init nested config
    optimizer = config.trainer.optimizer.init()
    assert isinstance(optimizer, Adam)
    assert optimizer.lr == 0.01

    # Init with override
    optimizer2 = config.trainer.optimizer.init(lr=0.001)
    assert optimizer2.lr == 0.001


# === Type Compatibility Tests ===


def test_parser_type_compatibility(tmp_path: Path):
    """Tests for: TYPE compatibility checking - valid subclass, invalid class, function return type."""
    # Invalid class check
    class WrongClass:
        pass

    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("optimizer:\n  TYPE: WrongClass")

    parser = SynConfParser()
    parser.add("optimizer", Optimizer)
    parser._resolver.register("WrongClass", WrongClass)

    with pytest.raises(TypeError, match="is not a subclass"):
        parser.parse(str(yaml_file))

    # Valid subclass
    yaml_file.write_text("optimizer:\n  TYPE: Adam\n  lr: 0.01")
    parser = SynConfParser()
    parser.add("optimizer", Optimizer)
    parser._resolver.register("Adam", Adam)
    config = parser.parse(str(yaml_file))  # Should not raise
    assert config.optimizer[TYPE_KEY] is Adam


# === Nested Type Validation Tests ===


def test_nested_type_validation():
    """Tests for: Nested TYPE validation - valid subclass, invalid class, function return, no type hint, deeply nested."""
    # Valid nested subclass
    class Model:
        def __init__(self, optimizer: Optimizer):
            self.optimizer = optimizer

    parser = SynConfParser()
    parser.add("model", Model)
    parser._resolver.register("Adam", Adam)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("model:\n  TYPE: Model\n  optimizer:\n    TYPE: Adam\n    lr: 0.01\n")

        config = parser.parse(config_path)
        model = config.model.init()
        assert isinstance(model.optimizer, Adam)

    # Invalid nested class
    parser = SynConfParser()
    parser.add("model", Model)
    parser._resolver.register("Scheduler", Scheduler)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("model:\n  TYPE: Model\n  optimizer:\n    TYPE: Scheduler\n")

        with pytest.raises(TypeError, match="is not a subclass"):
            parser.parse(config_path)

    # Function return type validation
    def build_optimizer(lr: float = 0.001) -> Optimizer:
        return Adam(lr=lr)

    class ModelWithFn:
        def __init__(self, optimizer: Optimizer):
            self.optimizer = optimizer

    parser = SynConfParser()
    parser.add("model", ModelWithFn)
    parser._resolver.register("build_optimizer", build_optimizer)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("model:\n  TYPE: ModelWithFn\n  optimizer:\n    TYPE: build_optimizer\n    lr: 0.01\n")

        config = parser.parse(config_path)
        model = config.model.init()
        assert isinstance(model.optimizer, Optimizer)

    # No type hint skips validation
    class ModelNoHint:
        def __init__(self, optimizer):  # No type hint
            self.optimizer = optimizer

    parser = SynConfParser()
    parser.add("model", ModelNoHint)
    parser._resolver.register("Scheduler", Scheduler)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("model:\n  TYPE: ModelNoHint\n  optimizer:\n    TYPE: Scheduler\n")

        config = parser.parse(config_path)  # Should not raise
        model = config.model.init()
        assert isinstance(model.optimizer, Scheduler)

    # Deeply nested validation
    class ModelWithScheduler:
        def __init__(self, optimizer: Optimizer):
            self.optimizer = optimizer

    class AdamWithScheduler(Optimizer):
        def __init__(self, scheduler: Scheduler = None, **kwargs):
            super().__init__(**kwargs)
            self.scheduler = scheduler

    parser = SynConfParser()
    parser.add("model", ModelWithScheduler)
    parser._resolver.register("AdamWithScheduler", AdamWithScheduler)
    parser._resolver.register("StepLR", StepLR)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(
                "model:\n"
                "  TYPE: ModelWithScheduler\n"
                "  optimizer:\n"
                "    TYPE: AdamWithScheduler\n"
                "    lr: 0.01\n"
                "    scheduler:\n"
                "      TYPE: StepLR\n"
                "      gamma: 0.5\n"
            )

        config = parser.parse(config_path)
        model = config.model.init()
        assert isinstance(model.optimizer, AdamWithScheduler)
        assert isinstance(model.optimizer.scheduler, StepLR)
        assert model.optimizer.scheduler.gamma == 0.5
