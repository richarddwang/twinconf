"""YAML file loader using PyYAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class YamlLoader:
    """Loads configuration from YAML files.

    Uses pyyaml's safe_load to parse YAML files into Python dictionaries.
    Values are automatically converted to appropriate Python types.
    """

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a YAML file and return its contents as a dictionary.

        Args:
            path: Path to the YAML file.

        Returns:
            Dictionary containing the parsed YAML content.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")

        with open(path, encoding="utf-8") as f:
            content = yaml.safe_load(f)

        # Handle empty YAML files
        if content is None:
            return {}

        # Ensure we return a dict
        if not isinstance(content, dict):
            raise ValueError(f"YAML file must contain a mapping at the root level, got {type(content).__name__}")

        return content

    def load_string(self, content: str) -> dict[str, Any]:
        """Parse a YAML string and return its contents as a dictionary.

        Args:
            content: YAML content as a string.

        Returns:
            Dictionary containing the parsed YAML content.

        Raises:
            yaml.YAMLError: If the string contains invalid YAML.
        """
        result = yaml.safe_load(content)

        # Handle empty YAML
        if result is None:
            return {}

        # Ensure we return a dict
        if not isinstance(result, dict):
            raise ValueError(f"YAML content must be a mapping at the root level, got {type(result).__name__}")

        return result


def is_yaml_file(source: str) -> bool:
    """Check if a source string is a YAML file path.

    Args:
        source: A source string (could be file path or CLI arg).

    Returns:
        True if the source appears to be a YAML file path.
    """
    # CLI args start with --
    if source.startswith("--"):
        return False

    # Check for YAML extensions
    lower = source.lower()
    return lower.endswith(".yaml") or lower.endswith(".yml")
