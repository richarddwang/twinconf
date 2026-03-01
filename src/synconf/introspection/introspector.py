"""Main Introspector class that orchestrates signature, docstring, and kwargs analysis."""

from __future__ import annotations

from typing import Any, Callable

from .docstring import extract_docstring
from .kwargs_tracer import KwargsTracer
from .signature import extract_signature, get_callable_name
from .spec import CallableSpec


class Introspector:
    """Introspects callables to extract complete parameter specifications.

    Combines signature extraction, docstring parsing, and kwargs tracing to build
    a complete CallableSpec with all parameters including those from kwargs delegees.
    """

    def __init__(self) -> None:
        """Initialize the Introspector with a kwargs tracer."""
        self._kwargs_tracer = KwargsTracer()
        self._cache: dict[int, CallableSpec] = {}  # id(callable) -> CallableSpec

    def analyze(self, target: type | Callable[..., Any]) -> CallableSpec:
        """Analyze a callable and return its complete specification.

        Args:
            target: A class, function, or method to introspect.

        Returns:
            CallableSpec containing all parameters including those from kwargs delegees.
        """
        # Check cache
        target_id = id(target)
        if target_id in self._cache:
            return self._cache[target_id]

        # Create the spec (add to cache early to handle circular references)
        spec = CallableSpec(
            name=get_callable_name(target),
            callable_obj=target,
        )
        self._cache[target_id] = spec

        # Extract signature
        params, has_var_keyword = extract_signature(target)

        # Extract docstring
        description, param_descriptions = extract_docstring(target)
        spec.description = description

        # Build ParamSpec objects with docstring descriptions
        for param in params:
            param.description = param_descriptions.get(param.name)
            param.source = spec
            spec.params[param.name] = param

        # Trace kwargs if present
        if has_var_keyword:
            delegees = self._kwargs_tracer.trace(target)
            for delegee in delegees:
                delegee_spec = self.analyze(delegee)
                spec.kwargs_delegees.append(delegee_spec)

        return spec

    def analyze_with_nested(
        self,
        target: type | Callable[..., Any],
        param_types: dict[str, type] | None = None,
    ) -> CallableSpec:
        """Analyze a callable including nested TYPE parameters.

        For parameters that are typed as classes (e.g., optimizer: Optimizer),
        also introspect those types to support nested configuration.

        Args:
            target: A class, function, or method to introspect.
            param_types: Optional mapping of parameter names to their expected types
                        for nested introspection.

        Returns:
            CallableSpec with nested parameter information.
        """
        spec = self.analyze(target)

        # Identify parameters that could have nested TYPE configs
        # These are parameters typed as classes (not primitives)
        if param_types is None:
            param_types = {}
            for param_name, param_spec in spec.get_all_params().items():
                hint = param_spec.type_hint
                if isinstance(hint, type) and not self._is_primitive_type(hint):
                    param_types[param_name] = hint

        # For each nested type, analyze it too
        for param_name, expected_type in param_types.items():
            if param_name in spec.params:
                # Mark that this param supports nested TYPE config
                # The actual nested spec will be created during resolution
                pass

        return spec

    def clear_cache(self) -> None:
        """Clear the introspection cache."""
        self._cache.clear()

    def _is_primitive_type(self, hint: Any) -> bool:
        """Check if a type hint is a primitive type.

        Args:
            hint: The type to check.

        Returns:
            True if the type is a primitive (str, int, float, bool, etc.).
        """
        primitives = (str, int, float, bool, bytes, type(None))
        return hint in primitives
