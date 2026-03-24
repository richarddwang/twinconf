"""
Tests for: REMOVE During Merging
Corresponds to: docs/advanced.md#remove-during-merging
"""

from pathlib import Path
from textwrap import dedent

from conftest import Model

from synconf import SynConfParser


def test_remove_during_merging(tmp_path: Path):
    """Tests for: REMOVE - YAML override removal and CLI removal in one merge."""
    base = tmp_path / "base.yaml"
    base.write_text(
        dedent("""\
        model:
          lr: 0.1
          peft_config:
            r: 2
        model2:
          lr: 0.2
          peft_config:
            r: 4
    """)
    )

    override = tmp_path / "override.yaml"
    override.write_text(
        dedent("""\
        model:
          peft_config: REMOVE
    """)
    )

    parser = SynConfParser()
    parser.add("model", Model)
    parser.add("model2", Model)

    # YAML REMOVE on model.peft_config, CLI REMOVE on model2.peft_config
    config = parser.parse(str(base), str(override), "--model2.peft_config=REMOVE")
    assert config.model.lr == 0.1
    assert config.model.to_dict().get("peft_config") is None
    assert config.model2.lr == 0.2
    assert config.model2.to_dict().get("peft_config") is None
