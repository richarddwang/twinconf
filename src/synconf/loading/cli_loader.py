"""CLI argument parser that converts --key.nested=value to nested dicts."""

from __future__ import annotations

from typing import Any

import yaml


class CliLoader:
    """Parses CLI arguments into nested dictionaries.

    Supports dotted key notation (--key.nested.value=x) and uses PyYAML
    to parse values, so YAML syntax works in CLI arguments.
    """

    def parse(self, arg: str) -> dict[str, Any]:
        """Parse a single CLI argument into a nested dictionary.

        Args:
            arg: A CLI argument in the format --key=value or --key.nested=value.

        Returns:
            Nested dictionary representing the argument.

        Raises:
            ValueError: If the argument format is invalid.
        """
        # Remove leading dashes
        if not arg.startswith("--"):
            raise ValueError(f"CLI argument must start with '--': {arg}")

        arg = arg[2:]

        # Split on first =
        if "=" not in arg:
            raise ValueError(f"CLI argument must contain '=': --{arg}")

        key_path, value_str = arg.split("=", 1)

        if not key_path:
            raise ValueError(f"CLI argument key cannot be empty: --{arg}")

        # Parse value using PyYAML (handles int, float, bool, list, dict, etc.)
        value = self._parse_value(value_str)

        # Convert dotted path to nested dict
        return self._build_nested_dict(key_path, value)

    def parse_many(self, args: list[str]) -> list[dict[str, Any]]:
        """Parse multiple CLI arguments into a list of nested dictionaries.

        Args:
            args: List of CLI arguments.

        Returns:
            List of nested dictionaries, one per argument.
        """
        return [self.parse(arg) for arg in args]

    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string using PyYAML.

        Args:
            value_str: The value part of a CLI argument.

        Returns:
            The parsed Python value.
        """
        # Use yaml.safe_load to parse the value
        # This handles: 0.01 -> float, 100 -> int, true -> bool, [1,2] -> list, etc.
        try:
            return yaml.safe_load(value_str)
        except yaml.YAMLError:
            # If YAML parsing fails, return as string
            return value_str

    def _build_nested_dict(self, key_path: str, value: Any) -> dict[str, Any]:
        """Build a nested dictionary from a dotted key path.

        Args:
            key_path: Dotted key path (e.g., "trainer.optimizer.lr").
            value: The value to set at the deepest level.

        Returns:
            Nested dictionary.
        """
        keys = key_path.split(".")
        result: dict[str, Any] = {}
        current = result

        # Build nested structure
        for i, key in enumerate(keys[:-1]):
            current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

        return result


def is_cli_arg(source: str) -> bool:
    """Check if a source string is a CLI argument.

    Args:
        source: A source string (could be file path or CLI arg).

    Returns:
        True if the source is a CLI argument (starts with --).
    """
    return source.startswith("--")
