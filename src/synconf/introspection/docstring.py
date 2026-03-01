"""Docstring parsing utilities using docstring-parser library."""

from __future__ import annotations

from typing import Any, Callable

from docstring_parser import DocstringStyle, parse


def extract_docstring(target: type | Callable[..., Any]) -> tuple[str | None, dict[str, str]]:
    """Extract description and parameter descriptions from a callable's docstring.

    Args:
        target: A class, function, or method to introspect.

    Returns:
        A tuple of (callable_description, param_descriptions) where param_descriptions
        maps parameter names to their descriptions.
    """
    # For classes, prefer class docstring over __init__ docstring for description
    # but use __init__ docstring for parameter descriptions
    if isinstance(target, type):
        # Get class description
        class_doc = target.__doc__
        init_doc = target.__init__.__doc__ if hasattr(target, "__init__") else None

        # Parse class docstring for description
        class_description = None
        if class_doc:
            parsed_class = parse(class_doc, style=DocstringStyle.GOOGLE)
            class_description = parsed_class.short_description

        # Parse __init__ docstring for parameters
        param_descriptions: dict[str, str] = {}
        if init_doc:
            parsed_init = parse(init_doc, style=DocstringStyle.GOOGLE)
            for param in parsed_init.params:
                if param.description:
                    param_descriptions[param.arg_name] = param.description

        # If no class description, try __init__ description
        if not class_description and init_doc:
            parsed_init = parse(init_doc, style=DocstringStyle.GOOGLE)
            class_description = parsed_init.short_description

        return class_description, param_descriptions

    # For functions/methods
    docstring = target.__doc__
    if not docstring:
        return None, {}

    parsed = parse(docstring, style=DocstringStyle.GOOGLE)

    description = parsed.short_description
    param_descriptions = {param.arg_name: param.description for param in parsed.params if param.description}

    return description, param_descriptions


def extract_full_docstring(target: type | Callable[..., Any]) -> dict[str, Any]:
    """Extract complete docstring information from a callable.

    Args:
        target: A class, function, or method to introspect.

    Returns:
        Dictionary containing short_description, long_description, params, returns, raises.
    """
    # Determine which docstring to use
    if isinstance(target, type):
        # For classes, prefer class docstring for description, __init__ for params
        docstring = target.__doc__ or (target.__init__.__doc__ if hasattr(target, "__init__") else None)
    else:
        docstring = target.__doc__

    if not docstring:
        return {
            "short_description": None,
            "long_description": None,
            "params": {},
            "returns": None,
            "raises": [],
        }

    parsed = parse(docstring, style=DocstringStyle.GOOGLE)

    return {
        "short_description": parsed.short_description,
        "long_description": parsed.long_description,
        "params": {
            p.arg_name: {
                "description": p.description,
                "type_name": p.type_name,
                "is_optional": p.is_optional,
                "default": p.default,
            }
            for p in parsed.params
        },
        "returns": {
            "description": parsed.returns.description if parsed.returns else None,
            "type_name": parsed.returns.type_name if parsed.returns else None,
        },
        "raises": [{"type": exc.type_name, "description": exc.description} for exc in parsed.raises],
    }
