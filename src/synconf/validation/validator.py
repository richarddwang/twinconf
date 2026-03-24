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
        source_map: dict[str, str] | None = None,
    ) -> list[str]:
        """Validate a configuration dictionary against a CallableSpec.

        Args:
            config: The configuration dictionary to validate.
            spec: The CallableSpec describing expected parameters.
            path: Current path in the config (for error messages).
            strict: If True, treat unknown parameters as errors.
            source_map: Optional mapping of key paths to their value sources (e.g., "base.yaml", "CLI").

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []
        all_params = spec.get_all_params()
        source_map = source_map or {}

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
                    metadata = self._format_metadata(
                        self._format_source(current_path, source_map),
                        self._format_owner(spec),
                    )
                    errors.append(f"Unexpected parameter: '{current_path}'{metadata}")
                continue

            param_spec = all_params[key]

            # Skip validation for nested configs (they have their own TYPE)
            if isinstance(value, dict) and TYPE_KEY in value:
                continue

            # Validate type
            if param_spec.type_hint is not None:
                if not self.validate(value, param_spec.type_hint):
                    metadata = self._format_metadata(
                        self._format_source(current_path, source_map),
                        self._format_owner(param_spec),
                    )
                    errors.append(
                        f"Invalid type for '{current_path}': expected {param_spec.format_type()}, "
                        f"got {type(value).__name__} {value!r}{metadata}"
                    )

        # Check for missing required parameters
        for param_name, param_spec in all_params.items():
            if param_spec.is_required and param_name not in config:
                current_path = f"{path}.{param_name}" if path else param_name
                metadata = self._format_metadata(
                    self._format_source(current_path, source_map),
                    self._format_owner(param_spec),
                )
                errors.append(f"Missing required parameter: '{current_path}'{metadata}")

        return errors

    def validate_standalone(
        self,
        config: dict[str, Any],
        standalone_args: dict[str, Any],
        group_keys: set[str],
        source_map: dict[str, str] | None = None,
    ) -> list[str]:
        """Validate top-level standalone arguments and flag unknown top-level keys.

        Only flags unknown top-level keys that don't belong to any group, standalone arg,
        or interpolation variable. Keys from YAML files are allowed (used for interpolation).
        Keys from CLI that match no group or standalone arg are flagged as unknown.

        Also checks missing required standalone args and type mismatches.

        Args:
            config: The full resolved configuration dictionary.
            standalone_args: Mapping of arg name to ParamSpec.
            group_keys: Set of registered group keys.
            source_map: Optional mapping of key paths to their value sources.

        Returns:
            List of validation error messages.
        """
        errors: list[str] = []
        source_map = source_map or {}

        # Only flag unknown top-level keys if they came from CLI
        for key in config:
            if key in group_keys or key in standalone_args:
                continue
            # Only flag as unknown if set via CLI (YAML keys are allowed for interpolation)
            if source_map.get(key) == "CLI":
                errors.append(f"Unexpected parameter: '{key}' [source: CLI]")

        # Validate standalone arg types and check for missing required
        for arg_name, param_spec in standalone_args.items():
            if arg_name in config:
                # Validate type
                if param_spec.type_hint is not None:
                    if not self.validate(config[arg_name], param_spec.type_hint):
                        metadata = self._format_metadata(
                            self._format_source(arg_name, source_map),
                        )
                        errors.append(
                            f"Invalid type for '{arg_name}': expected {param_spec.format_type()}, "
                            f"got {type(config[arg_name]).__name__} {config[arg_name]!r}{metadata}"
                        )
            elif param_spec.is_required:
                errors.append(f"Missing required parameter: '{arg_name}'")

        return errors

    def validate_or_raise(
        self,
        config: dict[str, Any],
        spec: CallableSpec,
        path: str = "",
        strict: bool = True,
        source_map: dict[str, str] | None = None,
    ) -> list[str]:
        """Validate a configuration and return errors (for collection by caller).

        Args:
            config: The configuration dictionary to validate.
            spec: The CallableSpec describing expected parameters.
            path: Current path in the config (for error messages).
            strict: If True, treat unknown parameters as errors.
            source_map: Optional mapping of key paths to their value sources.

        Returns:
            List of validation error messages (empty if valid).
        """
        return self.validate_config(config, spec, path, strict, source_map)

    def _format_source(self, key_path: str, source_map: dict[str, str]) -> str:
        """Format source metadata for where the value was last set.

        Args:
            key_path: The key path to look up.
            source_map: Mapping of key paths to sources.

        Returns:
            String like 'source: CLI' or empty string.
        """
        if key_path in source_map:
            return f"source: {source_map[key_path]}"
        return ""

    def _format_owner(self, spec_or_param: Any) -> str:
        """Format owner metadata for which class or function defines the parameter.

        Args:
            spec_or_param: A ParamSpec (uses .source) or CallableSpec (uses .name directly).

        Returns:
            String like 'owner: Adam' or empty string.
        """
        # CallableSpec has .callable_obj; ParamSpec has .source
        if hasattr(spec_or_param, "callable_obj"):
            return f"owner: {spec_or_param.name}"
        if hasattr(spec_or_param, "source") and spec_or_param.source:
            return f"owner: {spec_or_param.source.name}"
        return ""

    def _format_metadata(self, *parts: str) -> str:
        """Format metadata parts into a bracketed suffix.

        Args:
            *parts: Metadata strings (empty ones are filtered out).

        Returns:
            Formatted string like ' [source: CLI, owner: Adam]' or empty string.
        """
        non_empty = [p for p in parts if p]
        if non_empty:
            return " [" + ", ".join(non_empty) + "]"
        return ""
