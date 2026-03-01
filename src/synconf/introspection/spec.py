"""Specification dataclasses for introspected callables and parameters."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ParamSpec:
    """Specification for a single parameter.

    Attributes:
        name: Parameter name.
        type_hint: Type annotation for the parameter.
        default: Default value, or inspect.Parameter.empty if required.
        description: Description from docstring.
        source: The CallableSpec that defined this parameter.
    """

    name: str
    type_hint: Any = None
    default: Any = inspect.Parameter.empty
    description: str | None = None
    source: CallableSpec | None = None

    @property
    def is_required(self) -> bool:
        """Check if this parameter is required (no default value)."""
        return self.default is inspect.Parameter.empty

    def format_type(self) -> str:
        """Format the type hint as a readable string.

        Returns:
            Human-readable type string.
        """
        # Handle missing type hint
        if self.type_hint is None or self.type_hint is inspect.Parameter.empty:
            return "Any"

        # Handle string annotations
        if isinstance(self.type_hint, str):
            return self.type_hint

        # Handle typing generics and regular types
        if hasattr(self.type_hint, "__name__"):
            return self.type_hint.__name__

        # Fallback to string representation
        return str(self.type_hint).replace("typing.", "")


@dataclass
class CallableSpec:
    """Specification for a callable (class, function, or method).

    Attributes:
        name: Name of the callable.
        callable_obj: Reference to the actual callable.
        description: Description from docstring.
        params: Dictionary mapping parameter names to ParamSpec.
        kwargs_delegees: List of callables that receive **kwargs from this callable.
    """

    name: str
    callable_obj: Callable[..., Any]
    description: str | None = None
    params: dict[str, ParamSpec] = field(default_factory=dict)
    kwargs_delegees: list[CallableSpec] = field(default_factory=list)

    def get_all_params(self) -> dict[str, ParamSpec]:
        """Get all parameters including those from kwargs delegees.

        Returns:
            Dictionary of all parameter specs, with delegee params included.
        """
        all_params: dict[str, ParamSpec] = {}

        # Add params from kwargs delegees first (so this callable's params override)
        for delegee in self.kwargs_delegees:
            delegee_params = delegee.get_all_params()
            for name, param in delegee_params.items():
                if name not in all_params:
                    all_params[name] = param

        # Add this callable's own params (they take precedence)
        for name, param in self.params.items():
            all_params[name] = param

        return all_params

    def is_class(self) -> bool:
        """Check if the callable is a class.

        Returns:
            True if callable_obj is a class.
        """
        return isinstance(self.callable_obj, type)

    def is_function(self) -> bool:
        """Check if the callable is a function.

        Returns:
            True if callable_obj is a function.
        """
        return inspect.isfunction(self.callable_obj)

    def is_method(self) -> bool:
        """Check if the callable is a method.

        Returns:
            True if callable_obj is a method.
        """
        return inspect.ismethod(self.callable_obj)
