"""
Tests for: Config Instantiation
Corresponds to: docs/core-concepts.md#initialization
"""

from pathlib import Path
from textwrap import dedent

import pytest
from conftest import Adam, Callback, CallbackTrainer, LoggingCallback, Trainer

from synconf import Config, SynConfParser


def test_config_init_and_overwrites(tmp_path: Path):
    """Tests for: Config.init - nested TYPE, list TYPE (function and method), overwrites, and alias."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        dedent("""\
        callback_trainer:
          callbacks:
            TYPE: LIST
            logger:
              TYPE: conftest.create_logging_callback
              name: logger
              level: DEBUG
            base:
              TYPE: conftest.CallbackTrainer.create_callback
              name: base
        trainer:
          max_epochs: 100
          optimizer:
            TYPE: conftest.Adam
            lr: 0.01
    """)
    )

    parser = SynConfParser()
    parser.add("callback_trainer", CallbackTrainer)
    parser.add("trainer", Trainer)
    config = parser.parse(str(yaml_file))

    # init produces correct nested instances (function and method TYPEs)
    ct = config.callback_trainer.init()
    assert isinstance(ct, CallbackTrainer) and len(ct.callbacks) == 2
    assert isinstance(ct.callbacks[0], LoggingCallback) and ct.callbacks[0].level == "DEBUG"
    assert isinstance(ct.callbacks[1], Callback) and ct.callbacks[1].name == "base"

    # instantiate() alias works
    assert isinstance(config.callback_trainer.instantiate(), CallbackTrainer)

    # init with overwrites
    trainer = config.trainer.init(max_epochs=200)
    assert trainer.max_epochs == 200 and isinstance(trainer.optimizer, Adam)


def test_config_init_missing_type():
    """Tests for: Missing TYPE error with configuration key in message."""
    config = Config({"model": {"lr": 0.01}})
    with pytest.raises(ValueError) as exc_info:
        config.model.init()

    expected = "Cannot init 'model': no TYPE specified in config"
    assert str(exc_info.value) == expected
