"""Deep merge utility for combining configuration dictionaries."""

from __future__ import annotations

from typing import Any

# Special value indicating key should be removed during merge
REMOVE_KEY = "REMOVE"


class Merger:
    """Deep merges configuration dictionaries.

    Later values override earlier values. Nested dictionaries are merged recursively.
    """

    def merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: The base dictionary.
            override: The dictionary to merge on top (values take precedence).

        Special behavior:
            If override value is the string "REMOVE", the key is deleted from the result.

        Returns:
            A new dictionary with merged values.
        """
        result = base.copy()

        for key, value in override.items():
            # Check for REMOVE marker
            if value == REMOVE_KEY:
                # Remove the key from result if it exists
                result.pop(key, None)
            elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = self.merge(result[key], value)
            else:
                # Override the value
                result[key] = value

        return result

    def merge_all(self, configs: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge multiple dictionaries in order.

        Args:
            configs: List of dictionaries to merge (later overrides earlier).

        Returns:
            A single merged dictionary.
        """
        if not configs:
            return {}

        result = configs[0].copy()
        for config in configs[1:]:
            result = self.merge(result, config)

        return result


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Convenience function for deep merging two dictionaries.

    Args:
        base: The base dictionary.
        override: The dictionary to merge on top.

    Returns:
        A new dictionary with merged values.
    """
    return Merger().merge(base, override)
