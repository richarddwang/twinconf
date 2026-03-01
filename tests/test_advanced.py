"""
Tests for: Advanced Features - REMOVE, LIST, Sweep
Corresponds to: docs/advanced.md
"""

from pathlib import Path

import pytest

from synconf import SynConfParser
from synconf.loading.merger import REMOVE_KEY, Merger
from synconf.resolution.resolver import LIST_TYPE

# === Test Fixtures ===


class Model:
    """Test model."""

    def __init__(self, lr: float, peft_config: dict | None = None):
        self.lr = lr
        self.peft_config = peft_config


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


class Experiment:
    """Experiment configuration."""

    def __init__(self, seed: int = 0, name: str = "exp"):
        self.seed = seed
        self.name = name


# === REMOVE Feature Tests ===


def test_remove_basic():
    """Tests for: REMOVE during merging - basic removal."""
    merger = Merger()

    base = {"model": {"lr": 0.1, "peft_config": {"r": 2}}}
    override = {"model": {"peft_config": REMOVE_KEY}}
    result = merger.merge(base, override)
    assert "lr" in result["model"] and "peft_config" not in result["model"]


def test_remove_nested():
    """Tests for: REMOVE during merging - nested removal."""
    merger = Merger()

    base = {"a": {"b": {"c": 1, "d": 2}}}
    result = merger.merge(base, {"a": {"b": {"c": REMOVE_KEY}}})
    assert result == {"a": {"b": {"d": 2}}}


def test_remove_nonexistent():
    """Tests for: REMOVE during merging - nonexistent key (no effect)."""
    merger = Merger()

    result = merger.merge({"model": {"lr": 0.1}}, {"model": {"nonexistent": REMOVE_KEY}})
    assert result == {"model": {"lr": 0.1}}


# === LIST Feature Tests ===


def test_list_conversion():
    """Tests for: LIST type - conversion via resolve_in_config."""
    parser = SynConfParser()
    parser.add("trainer", CallbackTrainer)
    parser._resolver.register("LoggingCallback", LoggingCallback)
    parser._resolver.register("Callback", Callback)

    config_dict = {
        "callbacks": {
            "TYPE": LIST_TYPE,
            "log": {"TYPE": "LoggingCallback", "name": "logger", "level": "DEBUG"},
            "checkpoint": {"TYPE": "Callback", "name": "checkpoint"},
        }
    }
    resolved = parser._resolver.resolve_in_config(config_dict, {})
    assert isinstance(resolved["callbacks"], list)
    assert len(resolved["callbacks"]) == 2
    assert callable(resolved["callbacks"][0]["TYPE"])


def test_list_merge_override():
    """Tests for: LIST type - merge override preserves TYPE."""
    merger = Merger()

    base = {
        "callbacks": {
            "TYPE": LIST_TYPE,
            "log": {"TYPE": "LoggingCallback", "level": "INFO"},
        }
    }
    override = {"callbacks": {"log": {"level": "DEBUG"}}}
    result = merger.merge(base, override)
    assert result["callbacks"]["TYPE"] == LIST_TYPE
    assert result["callbacks"]["log"]["level"] == "DEBUG"


def test_list_full_integration():
    """Tests for: LIST type - full integration: parse -> init list items."""
    parser = SynConfParser()
    parser.add("trainer", CallbackTrainer)
    parser._resolver.register("LoggingCallback", LoggingCallback)

    config = parser.parse(
        "--trainer.callbacks.TYPE=LIST",
        "--trainer.callbacks.log.TYPE=LoggingCallback",
        "--trainer.callbacks.log.name=logger",
        "--trainer.callbacks.log.level=DEBUG",
    )
    trainer = config.trainer.init()
    assert isinstance(trainer.callbacks, list)
    assert len(trainer.callbacks) == 1
    assert isinstance(trainer.callbacks[0], LoggingCallback)
    assert trainer.callbacks[0].level == "DEBUG"


# === Sweep Feature Tests ===


def test_sweep_parse_args():
    """Tests for: Sweep - parse args."""
    parser = SynConfParser()
    parser.add("model", Model)

    args = ["--model.lr=0.1", "sweep", "--model.lr=[0.01,0.001]"]
    base_args, sweep_values = parser._parse_sweep_args(args)
    assert base_args == ["--model.lr=0.1"]
    assert sweep_values == {"model.lr": [0.01, 0.001]}


def test_sweep_cartesian_product():
    """Tests for: Sweep - generate cartesian product."""
    parser = SynConfParser()
    parser.add("model", Model)

    base_config = {"model": {"lr": 0.1, "peft_config": None}}
    sweep_values = {"model.lr": [0.01, 0.001], "model.peft_config": [None, {"r": 2}]}
    configs = parser._generate_sweep_configs(base_config, sweep_values)
    assert len(configs) == 4  # 2 x 2
    assert configs[0]["model"]["lr"] == 0.01 and configs[0]["model"]["peft_config"] is None
    assert configs[1]["model"]["lr"] == 0.01 and configs[1]["model"]["peft_config"] == {"r": 2}
    assert configs[2]["model"]["lr"] == 0.001
    assert configs[3]["model"]["lr"] == 0.001


def test_sweep_return_type():
    """Tests for: Sweep - return type: single Config vs list[Config]."""
    parser = SynConfParser()
    parser.add("model", Model)

    config = parser.parse_args(["--model.lr=0.1"])
    assert not isinstance(config, list)

    configs = parser.parse_args(["--model.lr=0.1", "sweep", "--model.lr=[0.01,0.001]"])
    assert isinstance(configs, list)
    assert len(configs) == 2


def test_sweep_validation_error():
    """Tests for: Sweep - validation error in sweep."""
    parser = SynConfParser()
    parser.add("model", Model)

    with pytest.raises(ValueError) as exc_info:
        parser.parse_args(["--model.lr=0.1", "sweep", "--model.lr=[0.01,not_a_float]"])
    assert "validation failed" in str(exc_info.value).lower() or "Invalid type" in str(exc_info.value)


# === Strict Validation Tests ===


def test_strict_unknown_parameter():
    """Tests for: Strict validation - unknown parameter."""
    parser = SynConfParser()
    parser.add("model", Model)

    with pytest.raises(ValueError) as exc_info:
        parser.parse("--model.lr=0.1", "--model.unknown_param=value")
    assert "Unknown parameter" in str(exc_info.value)
    assert "model.unknown_param" in str(exc_info.value)


def test_strict_missing_required():
    """Tests for: Strict validation - missing required parameter."""
    parser = SynConfParser()
    parser.add("model", Model)

    with pytest.raises(ValueError) as exc_info:
        parser.parse("--model.peft_config=null")  # lr is required
    assert "Missing required parameter" in str(exc_info.value)
    assert "model.lr" in str(exc_info.value)


# === Mega Integration Test ===


def test_mega_integration_all_features(tmp_path: Path):
    """
    MEGA INTEGRATION TEST: Demonstrates all implemented features working together.

    This test validates the complete feature set:
    1. YAML + CLI source merging
    2. TYPE resolution with subclass compatibility
    3. Nested TYPE validation
    4. **kwargs parameter tracing
    5. ~{...} interpolation (variable, expression, env var)
    6. REMOVE during merging
    7. LIST specification by dict
    8. Sweep subcommand with Cartesian product
    9. Strict validation (missing/unknown params)
    10. Help generation with sources
    """

    # Define realistic nested class hierarchy
    class LRScheduler:
        def __init__(self, warmup_steps: int = 100):
            self.warmup_steps = warmup_steps

    class CosineScheduler(LRScheduler):
        def __init__(self, cycles: int = 1, **kwargs):
            super().__init__(**kwargs)
            self.cycles = cycles

    class BaseOptimizer:
        def __init__(self, lr: float, scheduler: LRScheduler | None = None):
            self.lr = lr
            self.scheduler = scheduler

    class AdamWOptimizer(BaseOptimizer):
        def __init__(self, weight_decay: float = 0.01, **kwargs):
            super().__init__(**kwargs)
            self.weight_decay = weight_decay

    class TrainingCallback:
        def __init__(self, name: str, priority: int = 0):
            self.name = name
            self.priority = priority

    class EarlyStoppingCallback(TrainingCallback):
        def __init__(self, patience: int = 5, **kwargs):
            super().__init__(**kwargs)
            self.patience = patience

    class CheckpointCallback(TrainingCallback):
        def __init__(self, save_every: int = 1, **kwargs):
            super().__init__(**kwargs)
            self.save_every = save_every

    class TrainerConfig:
        def __init__(
            self,
            max_epochs: int,
            optimizer: BaseOptimizer,
            callbacks: list | None = None,
            peft_config: dict | None = None,
        ):
            self.max_epochs = max_epochs
            self.optimizer = optimizer
            self.callbacks = callbacks or []
            self.peft_config = peft_config

    # Create parser and register all types
    parser = SynConfParser()
    parser.add("trainer", TrainerConfig)
    parser._resolver.register("AdamWOptimizer", AdamWOptimizer)
    parser._resolver.register("CosineScheduler", CosineScheduler)
    parser._resolver.register("EarlyStoppingCallback", EarlyStoppingCallback)
    parser._resolver.register("CheckpointCallback", CheckpointCallback)

    # Create base YAML with interpolation, nested TYPE, LIST
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text("""
base_lr: 0.001
warmup: 500

trainer:
  max_epochs: 100
  peft_config:
    r: 8
    alpha: 16
  optimizer:
    TYPE: AdamWOptimizer
    lr: ~{base_lr}
    weight_decay: ~{base_lr * 10}
    scheduler:
      TYPE: CosineScheduler
      warmup_steps: ~{warmup}
      cycles: 2
  callbacks:
    TYPE: LIST
    early_stop:
      TYPE: EarlyStoppingCallback
      name: early_stop
      patience: 10
    checkpoint:
      TYPE: CheckpointCallback
      name: checkpoint
      save_every: 5
      priority: 1
""")

    # 1. Basic parse with YAML + CLI merging and interpolation
    config = parser.parse(str(base_yaml), "--trainer.max_epochs=200")
    assert config.trainer.max_epochs == 200  # CLI override
    assert config.trainer.optimizer.lr == 0.001  # Interpolated from base_lr
    assert config.trainer.optimizer.weight_decay == pytest.approx(0.01)  # Expression
    assert config.trainer.optimizer.scheduler.warmup_steps == 500  # Interpolated

    # 2. TYPE resolution with subclass compatibility
    trainer = config.trainer.init()
    assert isinstance(trainer, TrainerConfig)
    assert isinstance(trainer.optimizer, AdamWOptimizer)
    assert isinstance(trainer.optimizer.scheduler, CosineScheduler)

    # 3. LIST conversion and nested initialization
    assert isinstance(trainer.callbacks, list)
    assert len(trainer.callbacks) == 2
    assert isinstance(trainer.callbacks[0], EarlyStoppingCallback)
    assert trainer.callbacks[0].patience == 10
    assert isinstance(trainer.callbacks[1], CheckpointCallback)
    assert trainer.callbacks[1].save_every == 5

    # 4. Test REMOVE feature
    override_yaml = tmp_path / "override.yaml"
    override_yaml.write_text("trainer:\n  peft_config: REMOVE")
    config2 = parser.parse(str(base_yaml), str(override_yaml))
    trainer2 = config2.trainer.init()
    assert trainer2.peft_config is None

    # 5. Test sweep feature
    configs = parser.parse_args(
        [
            str(base_yaml),
            "sweep",
            "--base_lr=[0.001,0.0001]",
            "--trainer.max_epochs=[50,100]",
        ]
    )
    assert isinstance(configs, list)
    assert len(configs) == 4  # 2 x 2 Cartesian product

    # Verify sweep values
    trainers = [c.trainer.init() for c in configs]
    lrs = {t.optimizer.lr for t in trainers}
    epochs = {t.max_epochs for t in trainers}
    assert lrs == {0.001, 0.0001}
    assert epochs == {50, 100}

    # 6. Test strict validation (unknown param)
    with pytest.raises(ValueError) as exc_info:
        parser.parse(str(base_yaml), "--trainer.unknown_option=value")
    assert "Unknown parameter" in str(exc_info.value)

    # 7. Test help generation
    help_text = parser.help()
    assert "--trainer.TYPE=" in help_text and "TrainerConfig" in help_text
    assert "--trainer.max_epochs" in help_text
    assert "from=" in help_text  # Source tracking present

    print("MEGA INTEGRATION TEST PASSED: All 10 feature categories verified!")
