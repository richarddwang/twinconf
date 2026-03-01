"""HelpBuilder for generating help text from CallableSpecs."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from synconf.introspection.spec import CallableSpec, ParamSpec

if TYPE_CHECKING:
    from synconf.introspection import Introspector
    from synconf.resolution.resolver import Resolver


class HelpBuilder:
    """Builds help text from CallableSpecs.

    Generates formatted help in CLI-style two-column format:
    - First column: --dot.path=value
    - Second column: <type> description [metadata]
    """

    def __init__(self, introspector: Introspector | None = None, resolver: Resolver | None = None) -> None:
        """Initialize the HelpBuilder.

        Args:
            introspector: Optional introspector for analyzing nested types.
            resolver: Optional resolver for resolving TYPE values.
        """
        self._introspector = introspector
        self._resolver = resolver

    def build(
        self,
        groups: dict[str, CallableSpec],
        config_data: dict[str, Any] | None = None,
        key_path_filter: str | None = None,
    ) -> str:
        """Build help text for all registered groups.

        Args:
            groups: Dictionary mapping group keys to CallableSpecs.
            config_data: Optional current configuration values.
            key_path_filter: Optional key path to filter arguments (e.g., "optimizer" or "trainer.optimizer").

        Returns:
            Formatted help text.
        """
        if not groups:
            return "No configuration groups registered."

        if config_data is None:
            config_data = {}

        # Collect all rows (path, value, type, description, metadata)
        rows: list[tuple[str, str]] = []

        for i, (group_key, spec) in enumerate(groups.items()):
            # Add empty line separator between groups (but not before first group)
            if i > 0 and not key_path_filter:
                rows.append(("", ""))  # Empty row as separator

            group_config = config_data.get(group_key, {})
            rows.extend(
                self._collect_rows(group_key, spec, group_config, parent_path="", key_path_filter=key_path_filter)
            )

        if not rows:
            if key_path_filter:
                return f"No configuration parameters found under '{key_path_filter}'."
            return "No configuration parameters."

        # Format with column alignment
        return self._format_rows(rows)

    def _collect_rows(
        self,
        key: str,
        spec: CallableSpec,
        config_value: Any,
        parent_path: str,
        key_path_filter: str | None = None,
        param_spec: ParamSpec | None = None,
    ) -> list[tuple[str, str]]:
        """Collect all parameter rows for a config group.

        Args:
            key: The configuration key.
            spec: The CallableSpec for this config.
            config_value: Current configuration value.
            parent_path: Parent dotted path (for nested configs).
            key_path_filter: Optional key path to filter results.
            param_spec: Optional parameter spec for nested configs (to show source metadata).

        Returns:
            List of (column1, column2) tuples.
        """
        rows: list[tuple[str, str]] = []

        # Build the full path
        if key:
            full_path = f"{parent_path}.{key}" if parent_path else key
        else:
            # For nested expansion, key is empty and parent_path is the full path
            full_path = parent_path

        # Apply filtering: check if this path matches the filter
        if key_path_filter is not None:
            # Only include if full_path starts with or equals the filter
            if not (full_path == key_path_filter or full_path.startswith(key_path_filter + ".")):
                # Also check if filter is under this path (for nested exploration)
                if not key_path_filter.startswith(full_path + "."):
                    return rows

        # The group itself (TYPE line) - show as .TYPE
        type_name = spec.name
        value_str = self._format_value(type_name)
        type_path = f"{full_path}.TYPE"
        col1 = f"--{type_path}={value_str}"

        # Build TYPE line description
        if param_spec:
            # For nested configs, use the parameter info but replace type with actual TYPE
            parts: list[str] = []

            # Use the actual TYPE name instead of parameter type hint
            parts.append(f"<{type_name}>")

            # Description from parameter
            if param_spec.description:
                parts.append(param_spec.description)

            # Metadata from parameter
            metadata: list[str] = []
            if param_spec.is_required:
                metadata.append("required")
            else:
                default_str = self._format_value(param_spec.default)
                metadata.append(f"default={default_str}")

            if param_spec.source:
                metadata.append(f"from={param_spec.source.name}")

            if metadata:
                parts.append(f"[{', '.join(metadata)}]")

            col2 = " ".join(parts)
        else:
            # For top-level groups, use the spec description
            col2 = f"<{type_name}> {spec.description or ''}"

        # Only add TYPE line if it matches the filter
        should_show_type = False
        if key_path_filter is None:
            should_show_type = True
        elif full_path == key_path_filter or full_path.startswith(key_path_filter + "."):
            # Show TYPE if we're at or under the filtered path
            should_show_type = True

        if should_show_type:
            rows.append((col1, col2.strip()))

        # Get all parameters
        all_params = spec.get_all_params()

        # Get current config dict (if it's a dict)
        current_config = config_value if isinstance(config_value, dict) else {}

        # Process each parameter (sorted alphabetically for consistent output)
        for param_name in sorted(all_params.keys()):
            param = all_params[param_name]
            param_path = f"{full_path}.{param_name}"

            # Apply filtering for parameters
            if key_path_filter is not None:
                # Only include params that match the filter prefix
                if not (param_path == key_path_filter or param_path.startswith(key_path_filter + ".")):
                    continue

            # Get current value
            param_value = current_config.get(param_name, inspect.Parameter.empty)

            # Check if this parameter is a nested object config with TYPE
            if isinstance(param_value, dict) and "TYPE" in param_value and self._introspector and self._resolver:
                # This is a nested config - expand it recursively instead of showing as single line
                try:
                    type_value = param_value["TYPE"]
                    resolved_type = self._resolver.resolve(type_value)

                    # Introspect to get the spec
                    nested_spec = self._introspector.analyze(resolved_type)

                    # Recursively collect rows for the nested config
                    # Pass the param_spec so the TYPE line shows source metadata
                    nested_rows = self._collect_rows(
                        key="",  # Empty key since param_path already has the full path
                        spec=nested_spec,
                        config_value=param_value,
                        parent_path=param_path,  # param_path becomes the parent
                        key_path_filter=key_path_filter,
                        param_spec=param,  # Pass parameter info for metadata display
                    )
                    rows.extend(nested_rows)
                    continue  # Skip adding the parameter line itself
                except Exception:
                    # If resolution/introspection fails, fall back to simple display
                    value_str = self._format_value(param_value.get("TYPE", "???"))
            elif param_value is inspect.Parameter.empty:
                # Value not in config - use default or ???
                if param.is_required:
                    value_str = "???"
                else:
                    value_str = self._format_value(param.default)
            else:
                value_str = self._format_value(param_value)

            # Build columns
            col1 = f"--{param_path}={value_str}"
            col2 = self._format_param_info(param)
            rows.append((col1, col2))

        return rows

    def _format_param_info(self, param: ParamSpec) -> str:
        """Format the info column for a parameter.

        Args:
            param: The parameter spec.

        Returns:
            Formatted info string: <type> description [metadata]
        """
        parts: list[str] = []

        # Type
        type_str = param.format_type()
        parts.append(f"<{type_str}>")

        # Description
        if param.description:
            parts.append(param.description)

        # Metadata
        metadata: list[str] = []

        # Required or default
        if param.is_required:
            metadata.append("required")
        else:
            default_str = self._format_value(param.default)
            metadata.append(f"default={default_str}")

        # Source (from=X)
        if param.source:
            metadata.append(f"from={param.source.name}")

        # Add metadata if any
        if metadata:
            parts.append(f"[{', '.join(metadata)}]")

        return " ".join(parts)

    def _format_value(self, value: Any) -> str:
        """Format a value in YAML style.

        Args:
            value: The value to format.

        Returns:
            YAML-formatted string.
        """
        if value is inspect.Parameter.empty:
            return "???"

        if value is None:
            return "null"

        if value is True:
            return "true"

        if value is False:
            return "false"

        if isinstance(value, str):
            # No quotes for strings in YAML style
            return value

        if isinstance(value, (int, float)):
            return str(value)

        if isinstance(value, list):
            if not value:
                return "[]"
            # Simple list representation
            return "[" + ", ".join(self._format_value(v) for v in value) + "]"

        if isinstance(value, dict):
            if not value:
                return "{}"
            # For dicts, just show {}
            return "{...}"

        # For classes/types, show the name
        if isinstance(value, type):
            return value.__name__

        # For other objects, try to get a reasonable representation
        if hasattr(value, "__name__"):
            return value.__name__

        return str(value)

    def _format_rows(self, rows: list[tuple[str, str]]) -> list[str]:
        """Format rows with column alignment.

        Args:
            rows: List of (col1, col2) tuples.

        Returns:
            Formatted lines with aligned columns.
        """
        if not rows:
            return []

        # Calculate max width of first column (ignoring empty separator rows)
        max_col1_width = max((len(row[0]) for row in rows if row[0]), default=0)

        # Format each row with padding
        lines: list[str] = []
        padding = 4  # Space between columns

        for col1, col2 in rows:
            # Handle empty separator rows
            if not col1 and not col2:
                lines.append("")
            else:
                # Pad first column to max width
                padded_col1 = col1.ljust(max_col1_width + padding)
                lines.append(f"{padded_col1}{col2}")

        return "\n".join(lines)
