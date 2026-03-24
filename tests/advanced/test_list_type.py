"""
Tests for: LIST Specification
Corresponds to: docs/advanced.md#list-specification
"""

from pathlib import Path
from textwrap import dedent

from conftest import Callback, CallbackTrainer, LoggingCallback

from synconf import SynConfParser


def test_list_type(tmp_path: Path):
    """Tests for: LIST type - YAML dict-to-list conversion via TYPE: LIST, init list items."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        dedent("""\
        trainer:
          callbacks:
            TYPE: LIST
            log:
              TYPE: conftest.LoggingCallback
              name: logger
              level: DEBUG
            checkpoint:
              TYPE: conftest.Callback
              name: checkpoint
    """)
    )

    parser = SynConfParser()
    parser.add("trainer", CallbackTrainer)
    config = parser.parse(str(yaml_file))

    trainer = config.trainer.init()
    assert isinstance(trainer.callbacks, list) and len(trainer.callbacks) == 2
    assert isinstance(trainer.callbacks[0], LoggingCallback) and trainer.callbacks[0].level == "DEBUG"
    assert isinstance(trainer.callbacks[1], Callback) and trainer.callbacks[1].name == "checkpoint"
