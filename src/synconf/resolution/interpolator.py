"""Interpolation resolution for ~{...} syntax."""

from __future__ import annotations

import ast
import os
import re
from enum import Enum
from typing import Any

import yaml

# Pattern to match ~{...} interpolations
INTERPOLATION_PATTERN = re.compile(r"~\{([^}]+)\}")

# Pattern to detect simple variable references (dotted path only, no dunder)
# Each segment must start with letter/underscore, not contain double underscores
REFERENCE_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*$")

# Environment variable prefix
ENV_PREFIX = "env:"

# Safe builtins for expression evaluation
SAFE_BUILTINS = {
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sum": sum,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "True": True,
    "False": False,
    "None": None,
}


class DotAccessDict:
    """Wrapper that allows dot-style attribute access on dict values.

    Used in expression evaluation to allow `model.layers` syntax.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize with dict data.

        Args:
            data: Dictionary to wrap.
        """
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        """Get value by attribute access.

        Args:
            name: Key name.

        Returns:
            Value, wrapped in DotAccessDict if it's a dict.
        """
        if name.startswith("_"):
            raise AttributeError(name)

        try:
            value = self._data[name]
            # Recursively wrap nested dicts
            if isinstance(value, dict):
                return DotAccessDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DotAccessDict({self._data!r})"


class InterpolationType(Enum):
    """Type of interpolation content."""

    REFERENCE = "reference"  # Simple dotted path: optimizer.lr
    EXPRESSION = "expression"  # Python expression: base_lr * 10
    ENV_VAR = "env_var"  # Environment variable: env:HOME


class InterpolationError(Exception):
    """Base exception for interpolation errors."""

    pass


class CircularInterpolationError(InterpolationError):
    """Raised when circular reference is detected."""

    def __init__(self, cycle_path: list[str]) -> None:
        """Initialize with the cycle path.

        Args:
            cycle_path: List of paths forming the cycle, e.g., ["a", "b.c", "a"].
        """
        cycle_str = " -> ".join(cycle_path)
        super().__init__(f"Circular reference detected: {cycle_str}")
        self.cycle_path = cycle_path


class UndefinedReferenceError(InterpolationError):
    """Raised when referencing undefined config key."""

    def __init__(self, path: str, available_keys: list[str] | None = None) -> None:
        """Initialize with the undefined path.

        Args:
            path: The dotted path that was not found.
            available_keys: Optional list of available keys for suggestion.
        """
        msg = f"Undefined reference: '{path}'"
        if available_keys:
            msg += f". Available keys: {', '.join(available_keys[:5])}"
        super().__init__(msg)
        self.path = path


class ExpressionError(InterpolationError):
    """Raised when expression is invalid or unsafe."""

    def __init__(self, expr: str, reason: str) -> None:
        """Initialize with expression and reason.

        Args:
            expr: The expression that failed.
            reason: Why it failed.
        """
        super().__init__(f"Invalid expression '~{{{expr}}}': {reason}")
        self.expr = expr
        self.reason = reason


class Interpolator:
    """Resolves ~{...} interpolations in configuration dictionaries.

    Supports three modes:
    - Variable reference: ~{path.to.value} - returns value at path, preserves type
    - Partial/embedded: prefix_~{var}_suffix - substitutes then re-parses as YAML
    - Expression: ~{base_lr * 10} - evaluates Python expression

    Environment variables: ~{env:HOME} - looks up os.environ

    Circular reference detection is performed during resolution.
    """

    def __init__(self) -> None:
        """Initialize the Interpolator."""
        # Stack to track currently resolving paths for cycle detection
        self._resolution_stack: list[str] = []
        # Cache for resolved values to avoid re-computation
        self._resolved_cache: dict[str, Any] = {}

    def resolve_all(self, config: dict[str, Any]) -> dict[str, Any]:
        """Resolve all ~{...} interpolations in a config dictionary.

        Args:
            config: The configuration dictionary to process.

        Returns:
            A new config dict with all interpolations resolved.

        Raises:
            CircularInterpolationError: If circular reference is detected.
            UndefinedReferenceError: If referencing nonexistent key.
            ExpressionError: If expression is invalid or unsafe.
        """
        # Reset state for fresh resolution
        self._resolution_stack = []
        self._resolved_cache = {}

        # Deep copy to avoid mutating input
        result = self._deep_copy(config)

        # Resolve all values recursively
        self._resolve_dict(result, result, "")

        return result

    def _resolve_dict(self, node: dict[str, Any], root: dict[str, Any], path_prefix: str) -> None:
        """Recursively resolve all string values in a dict.

        Args:
            node: Current dict node to process.
            root: Root config dict for lookups.
            path_prefix: Dotted path prefix to current node.
        """
        for key, value in node.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key

            if isinstance(value, str) and INTERPOLATION_PATTERN.search(value):
                # String with interpolation - resolve it
                node[key] = self._resolve_string(value, root, current_path)

            elif isinstance(value, dict):
                # Recurse into nested dict
                self._resolve_dict(value, root, current_path)

            elif isinstance(value, list):
                # Process list items
                self._resolve_list(value, root, current_path)

    def _resolve_list(self, items: list[Any], root: dict[str, Any], path_prefix: str) -> None:
        """Resolve interpolations in list items.

        Args:
            items: List to process.
            root: Root config dict for lookups.
            path_prefix: Dotted path prefix.
        """
        for i, item in enumerate(items):
            current_path = f"{path_prefix}[{i}]"

            if isinstance(item, str) and INTERPOLATION_PATTERN.search(item):
                items[i] = self._resolve_string(item, root, current_path)

            elif isinstance(item, dict):
                self._resolve_dict(item, root, current_path)

            elif isinstance(item, list):
                self._resolve_list(item, root, current_path)

    def _resolve_string(self, value: str, root: dict[str, Any], path: str) -> Any:
        """Resolve all ~{...} in a string value.

        Args:
            value: String containing interpolations.
            root: Root config dict for lookups.
            path: Dotted path of this value (for error messages).

        Returns:
            Resolved value - original type if single complete interpolation,
            YAML-reparsed value if partial/multiple interpolations.
        """
        matches = list(INTERPOLATION_PATTERN.finditer(value))

        if not matches:
            return value

        # Check if single interpolation covers entire string
        if len(matches) == 1 and matches[0].group(0) == value:
            # Complete replacement - preserve original type
            content = matches[0].group(1)
            return self._resolve_interpolation(content, root, path)

        # Partial substitution - build result string then reparse as YAML
        result = value
        for match in reversed(matches):  # Reverse to preserve indices
            content = match.group(1)
            resolved = self._resolve_interpolation(content, root, path)
            # Convert to YAML string representation
            resolved_str = self._to_yaml_string(resolved)
            result = result[: match.start()] + resolved_str + result[match.end() :]

        # Re-parse the resulting string as YAML scalar
        return self._parse_yaml_scalar(result)

    def _resolve_interpolation(self, content: str, root: dict[str, Any], path: str) -> Any:
        """Resolve a single interpolation content.

        Args:
            content: Content inside ~{...}.
            root: Root config dict.
            path: Path of the value being resolved (for error context).

        Returns:
            Resolved value.
        """
        interp_type = self._classify(content)

        if interp_type == InterpolationType.ENV_VAR:
            # Environment variable lookup
            env_name = content[len(ENV_PREFIX) :]
            return self._resolve_env(env_name)

        elif interp_type == InterpolationType.REFERENCE:
            # Simple variable reference
            return self._resolve_reference(content, root, path)

        else:
            # Python expression
            return self._resolve_expression(content, root, path)

    def _classify(self, content: str) -> InterpolationType:
        """Classify interpolation content type.

        Args:
            content: Content inside ~{...}.

        Returns:
            InterpolationType indicating how to process it.
        """
        # Check for environment variable
        if content.startswith(ENV_PREFIX):
            return InterpolationType.ENV_VAR

        # Check for dunder attributes (security risk)
        if "__" in content:
            return InterpolationType.EXPRESSION

        # Check for simple reference (dotted path only)
        if REFERENCE_PATTERN.match(content):
            return InterpolationType.REFERENCE

        # Otherwise it's an expression
        return InterpolationType.EXPRESSION

    def _resolve_reference(self, dotted_path: str, root: dict[str, Any], from_path: str) -> Any:
        """Resolve a dotted path reference.

        Args:
            dotted_path: Path like "optimizer.lr".
            root: Root config dict.
            from_path: Path of value requesting this reference (for cycle detection).

        Returns:
            Value at the path.
        """
        # Check for circular reference
        if dotted_path in self._resolution_stack:
            cycle = self._resolution_stack[self._resolution_stack.index(dotted_path) :] + [dotted_path]
            raise CircularInterpolationError(cycle)

        # Check cache first
        if dotted_path in self._resolved_cache:
            return self._resolved_cache[dotted_path]

        # Add to resolution stack
        self._resolution_stack.append(dotted_path)

        try:
            # Navigate to the value
            parts = dotted_path.split(".")
            current = root

            for i, part in enumerate(parts):
                if not isinstance(current, dict):
                    partial_path = ".".join(parts[: i + 1])
                    raise UndefinedReferenceError(dotted_path, available_keys=list(root.keys()) if i == 0 else None)

                if part not in current:
                    partial_path = ".".join(parts[: i + 1])
                    raise UndefinedReferenceError(dotted_path, available_keys=list(current.keys()))

                current = current[part]

            # If the value itself contains interpolations, resolve them first
            if isinstance(current, str) and INTERPOLATION_PATTERN.search(current):
                current = self._resolve_string(current, root, dotted_path)
                # Update the root config with resolved value
                self._set_value(root, dotted_path, current)

            # Cache the resolved value
            self._resolved_cache[dotted_path] = current

            return current

        finally:
            # Remove from resolution stack
            self._resolution_stack.pop()

    def _resolve_expression(self, expr: str, root: dict[str, Any], path: str) -> Any:
        """Evaluate a Python expression with config values.

        Args:
            expr: Python expression like "base_lr * 10".
            root: Root config dict.
            path: Path of value being resolved (for error context).

        Returns:
            Expression result.
        """
        # Validate expression safety
        self._validate_expression(expr)

        # Build namespace from config
        namespace = self._build_namespace(root, path)

        # Evaluate expression
        try:
            code = compile(expr, "<interpolation>", "eval")
            return eval(code, {"__builtins__": SAFE_BUILTINS}, namespace)
        except NameError as e:
            # Extract variable name from error
            raise ExpressionError(expr, str(e))
        except Exception as e:
            raise ExpressionError(expr, str(e))

    def _validate_expression(self, expr: str) -> None:
        """Validate that expression is safe to evaluate.

        Args:
            expr: Expression to validate.

        Raises:
            ExpressionError: If expression contains unsafe constructs.
        """
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ExpressionError(expr, f"syntax error: {e.msg}")

        # Walk AST and check for disallowed constructs
        for node in ast.walk(tree):
            # Disallow imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ExpressionError(expr, "imports not allowed")

            # Disallow lambda
            if isinstance(node, ast.Lambda):
                raise ExpressionError(expr, "lambda not allowed")

            # Disallow dunder attributes
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ExpressionError(expr, f"access to '{node.attr}' not allowed")

            # Disallow dunder names
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise ExpressionError(expr, f"access to '{node.id}' not allowed")

    def _build_namespace(self, root: dict[str, Any], from_path: str) -> dict[str, Any]:
        """Build namespace for expression evaluation.

        Flattens config with both dotted access and underscore-joined keys.
        Only resolves reference-type interpolations to avoid infinite recursion.

        Args:
            root: Root config dict.
            from_path: Path of value requesting evaluation.

        Returns:
            Namespace dict for eval.
        """
        namespace: dict[str, Any] = {}

        # Add top-level keys directly
        for key, value in root.items():
            # Only resolve reference-type interpolations (not expressions)
            if isinstance(value, str) and INTERPOLATION_PATTERN.search(value):
                value = self._resolve_references_only(value, root, key)

            # Wrap dicts in DotAccessDict for dot-style attribute access in expressions
            if isinstance(value, dict):
                namespace[key] = DotAccessDict(value)
                # Also add flattened access with double underscore
                self._flatten_to_namespace(value, key, namespace, root)
            else:
                namespace[key] = value

        return namespace

    def _flatten_to_namespace(
        self,
        node: dict[str, Any],
        prefix: str,
        namespace: dict[str, Any],
        root: dict[str, Any],
    ) -> None:
        """Recursively flatten dict to namespace with underscore-joined keys.

        Args:
            node: Dict to flatten.
            prefix: Key prefix.
            namespace: Namespace being built.
            root: Root config for resolving references.
        """
        for key, value in node.items():
            full_key = f"{prefix}__{key}"

            # Only resolve reference-type interpolations (not expressions)
            if isinstance(value, str) and INTERPOLATION_PATTERN.search(value):
                value = self._resolve_references_only(value, root, f"{prefix}.{key}")

            namespace[full_key] = value

            if isinstance(value, dict):
                self._flatten_to_namespace(value, full_key, namespace, root)

    def _resolve_references_only(self, value: str, root: dict[str, Any], path: str) -> Any:
        """Resolve only reference-type interpolations, skip expressions.

        Used by namespace building to avoid infinite recursion.

        Args:
            value: String containing interpolations.
            root: Root config dict.
            path: Dotted path of this value.

        Returns:
            Resolved value if it's a simple reference, original otherwise.
        """
        matches = list(INTERPOLATION_PATTERN.finditer(value))

        if not matches:
            return value

        # Only resolve if single interpolation covers entire string AND is a reference
        if len(matches) == 1 and matches[0].group(0) == value:
            content = matches[0].group(1)
            interp_type = self._classify(content)

            # Only resolve references, not expressions or env vars (during namespace build)
            if interp_type == InterpolationType.REFERENCE:
                return self._resolve_reference(content, root, path)

        # Can't fully resolve - return original value
        return value

    def _resolve_env(self, name: str) -> str:
        """Resolve environment variable.

        Args:
            name: Environment variable name.

        Returns:
            Environment variable value.

        Raises:
            UndefinedReferenceError: If environment variable not set.
        """
        value = os.environ.get(name)
        if value is None:
            raise UndefinedReferenceError(f"env:{name}")
        return value

    def _to_yaml_string(self, value: Any) -> str:
        """Convert value to YAML string representation for embedding.

        Args:
            value: Value to convert.

        Returns:
            YAML string representation.
        """
        if value is None:
            return "null"

        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, (int, float, str)):
            return str(value)

        # For complex types, use YAML dump
        return yaml.safe_dump(value, default_flow_style=True).strip()

    def _parse_yaml_scalar(self, s: str) -> Any:
        """Re-parse a string as YAML scalar value.

        Args:
            s: String to parse.

        Returns:
            Parsed YAML value.
        """
        try:
            return yaml.safe_load(s)
        except yaml.YAMLError:
            # If parsing fails, return as-is
            return s

    def _set_value(self, root: dict[str, Any], dotted_path: str, value: Any) -> None:
        """Set a value at a dotted path in the config.

        Args:
            root: Root config dict.
            dotted_path: Path like "optimizer.lr".
            value: Value to set.
        """
        parts = dotted_path.split(".")
        current = root

        for part in parts[:-1]:
            current = current[part]

        current[parts[-1]] = value

    def _deep_copy(self, obj: Any) -> Any:
        """Create a deep copy of a nested structure.

        Args:
            obj: Object to copy.

        Returns:
            Deep copy.
        """
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]

        # Primitives and other types are returned as-is
        return obj
