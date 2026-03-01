"""Type validation using beartype."""

from __future__ import annotations

from typing import Any

from beartype.door import is_bearable

from synconf.introspection import CallableSpec
from synconf.resolution.resolver import TYPE_KEY


class Validator:
    """Validates configuration values against type hints.

    Uses beartype's is_bearable for efficient runtime type checking.
    """

    def validate(self, value: Any, type_hint: Any) -> bool:
        """Check if a value matches a type hint.

        Args:
            value: The value to validate.
            type_hint: The expected type hint.

        Returns:
            True if the value matches the type hint.
        """
        # Handle None/missing type hint
        if type_hint is None:
            return True

        # Use beartype for validation
        return is_bearable(value, type_hint)

    def validate_config(
        self,
        config: dict[str, Any],
        spec: CallableSpec,
        path: str = "",
        strict: bool = True,
    ) -> list[str]:
        """Validate a configuration dictionary against a CallableSpec.

        Args:
            config: The configuration dictionary to validate.
            spec: The CallableSpec describing expected parameters.
            path: Current path in the config (for error messages).
            strict: If True, treat unknown parameters as errors.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []
        all_params = spec.get_all_params()

        for key, value in config.items():
            # Skip TYPE key
            if key == TYPE_KEY:
                continue

            current_path = f"{path}.{key}" if path else key

            # Check if parameter is known
            if key not in all_params:
                # Check if it's a nested config with TYPE (allowed)
                if isinstance(value, dict) and TYPE_KEY in value:
                    continue

                # Unknown parameter
                if strict:
                    errors.append(f"Unknown parameter: '{current_path}'")
                continue

            param_spec = all_params[key]

            # Skip validation for nested configs (they have their own TYPE)
            if isinstance(value, dict) and TYPE_KEY in value:
                continue

            # Validate type
            if param_spec.type_hint is not None:
                if not self.validate(value, param_spec.type_hint):
                    errors.append(
                        f"Invalid type for '{current_path}': expected {param_spec.format_type()}, "
                        f"got {type(value).__name__} ({value!r})"
                    )

        # Check for missing required parameters
        for param_name, param_spec in all_params.items():
            if param_spec.is_required and param_name not in config:
                current_path = f"{path}.{param_name}" if path else param_name
                errors.append(f"Missing required parameter: '{current_path}'")

        return errors

    def validate_or_raise(
        self,
        config: dict[str, Any],
        spec: CallableSpec,
        path: str = "",
        strict: bool = True,
    ) -> None:
        """Validate a configuration and raise if invalid.

        Args:
            config: The configuration dictionary to validate.
            spec: The CallableSpec describing expected parameters.
            path: Current path in the config (for error messages).
            strict: If True, treat unknown parameters as errors.

        Raises:
            ValueError: If validation fails, with all error messages.
        """
        errors = self.validate_config(config, spec, path, strict)
        if errors:
            raise ValueError("Configuration validation failed:\n  " + "\n  ".join(errors))
