"""
Tests for: Help Generation
Corresponds to: docs/core-concepts.md#help-generation
"""

from pathlib import Path
from textwrap import dedent

import pytest
from conftest import Adam, Trainer

from synconf import SynConfParser


def test_help_generation(tmp_path: Path, capsys):
    """Tests for: Help - group output, kwargs sources, required, defaults, config overrides, nested TYPE expansion."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        dedent("""\
        optimizer:
          lr: 0.001
          beta1: 0.95
        trainer:
          max_epochs: 100
          optimizer:
            TYPE: conftest.Adam
            lr: 0.01
    """)
    )

    parser = SynConfParser()
    parser.add("optimizer", Adam)
    parser.add("trainer", Trainer)

    with pytest.raises(SystemExit):
        parser.parse(str(yaml_file), "--help")

    expected = dedent("""\
        --optimizer.TYPE=Adam                   <Adam> Adam optimizer extends BaseOptimizer.
        --optimizer.beta1=0.95                  <float> First moment decay. [default=0.9, from=Adam]
        --optimizer.beta2=0.999                 <float> Second moment decay. [default=0.999, from=Adam]
        --optimizer.lr=0.001                    <float> Learning rate. [required, from=BaseOptimizer]
        --optimizer.weight_decay=0.0            <float> Weight decay factor. [default=0.0, from=BaseOptimizer]

        --trainer.TYPE=Trainer                  <Trainer> Training controller.
        --trainer.max_epochs=100                <int> Maximum training epochs. [required, from=Trainer]
        --trainer.optimizer.TYPE=Adam           <Adam> Optimizer instance. [default=null, from=Trainer]
        --trainer.optimizer.beta1=0.9           <float> First moment decay. [default=0.9, from=Adam]
        --trainer.optimizer.beta2=0.999         <float> Second moment decay. [default=0.999, from=Adam]
        --trainer.optimizer.lr=0.01             <float> Learning rate. [required, from=BaseOptimizer]
        --trainer.optimizer.weight_decay=0.0    <float> Weight decay factor. [default=0.0, from=BaseOptimizer]
    """)
    assert capsys.readouterr().out == expected


def test_help_scoped(tmp_path: Path, capsys):
    """Tests for: Scoped help - --help=<key> shows only the corresponding section."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        dedent("""\
        optimizer:
          lr: 0.001
          beta1: 0.95
        trainer:
          max_epochs: 100
          optimizer:
            TYPE: conftest.Adam
            lr: 0.01
    """)
    )

    parser = SynConfParser()
    parser.add("optimizer", Adam)
    parser.add("trainer", Trainer)

    # --help=optimizer: shows only the top-level optimizer group
    with pytest.raises(SystemExit):
        parser.parse(str(yaml_file), "--help=optimizer")

    expected_optimizer = dedent("""\
        --optimizer.TYPE=Adam           <Adam> Adam optimizer extends BaseOptimizer.
        --optimizer.beta1=0.95          <float> First moment decay. [default=0.9, from=Adam]
        --optimizer.beta2=0.999         <float> Second moment decay. [default=0.999, from=Adam]
        --optimizer.lr=0.001            <float> Learning rate. [required, from=BaseOptimizer]
        --optimizer.weight_decay=0.0    <float> Weight decay factor. [default=0.0, from=BaseOptimizer]
    """)
    assert capsys.readouterr().out == expected_optimizer
