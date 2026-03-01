"""Signature extraction utilities using inspect module."""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

from .spec import ParamSpec


def extract_signature(target: type | Callable[..., Any]) -> tuple[list[ParamSpec], bool]:
    """Extract parameter specifications from a callable's signature.

    Args:
        target: A class, function, or method to introspect.

    Returns:
        A tuple of (list of ParamSpec, has_var_keyword) where has_var_keyword indicates
        whether the callable accepts **kwargs.
    """
    # For classes, analyze __init__
    if isinstance(target, type):
        func = target.__init__
    else:
        func = target

    # Get signature
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return [], False

    # Get type hints (resolves forward references)
    try:
        hints = get_type_hints(func)
    except Exception:
        # May fail with forward refs or missing imports
        hints = {}

    params: list[ParamSpec] = []
    has_var_keyword = False

    for name, param in sig.parameters.items():
        # Skip self/cls
        if name in ("self", "cls"):
            continue

        # Track if there's **kwargs
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue

        # Skip *args
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue

        # Get type hint from typing.get_type_hints or fallback to annotation
        type_hint = hints.get(name, param.annotation)
        if type_hint is inspect.Parameter.empty:
            type_hint = None

        params.append(
            ParamSpec(
                name=name,
                type_hint=type_hint,
                default=param.default,
                description=None,  # Filled in by docstring extraction
                source=None,  # Filled in by Introspector
            )
        )

    return params, has_var_keyword


def get_callable_name(target: type | Callable[..., Any]) -> str:
    """Get the qualified name of a callable.

    Args:
        target: A class, function, or method.

    Returns:
        The qualified name (e.g., 'ClassName' or 'module.func').
    """
    if hasattr(target, "__qualname__"):
        return target.__qualname__
    if hasattr(target, "__name__"):
        return target.__name__
    return str(target)


def get_callable_for_introspection(target: type | Callable[..., Any]) -> Callable[..., Any]:
    """Get the actual function to introspect for a given target.

    For classes, returns __init__. For methods/functions, unwraps decorators.

    Args:
        target: A class, function, or method.

    Returns:
        The function to introspect.
    """
    if isinstance(target, type):
        return target.__init__

    # Unwrap decorated functions
    return inspect.unwrap(target)


def get_return_type(target: Callable[..., Any]) -> Any | None:
    """Get the return type annotation of a callable.

    Args:
        target: A function or method.

    Returns:
        The return type annotation, or None if not annotated.
    """
    try:
        hints = get_type_hints(target)
        return hints.get("return")
    except Exception:
        # Fallback to direct annotation
        sig = inspect.signature(target)
        if sig.return_annotation is not inspect.Parameter.empty:
            return sig.return_annotation
        return None
