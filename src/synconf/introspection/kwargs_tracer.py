"""AST-based kwargs tracing to find callees that receive **kwargs."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Callable


class KwargsTracer:
    """Traces **kwargs delegation through callable bodies using AST analysis.

    This tracer analyzes the source code of a callable to find where **kwargs
    is passed to other callables (e.g., super().__init__(**kwargs)).
    """

    def trace(self, target: type | Callable[..., Any]) -> list[Callable[..., Any]]:
        """Trace all callables that receive **kwargs from the target.

        Args:
            target: A class, function, or method to analyze.

        Returns:
            List of callable objects that receive **kwargs from the target.
        """
        # Get the function to analyze
        if isinstance(target, type):
            func = target.__init__
            context_class = target
        else:
            func = target
            context_class = None

        # Get source code
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            # Cannot get source (e.g., C extension)
            return []

        # Dedent source to handle nested definitions
        source = textwrap.dedent(source)

        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        # Find the function definition
        func_def = self._find_function_def(tree, func.__name__)
        if func_def is None:
            return []

        # Get the **kwargs parameter name
        kwarg_name = self._get_kwarg_param_name(func_def)
        if kwarg_name is None:
            return []

        # Find all calls that receive **kwarg_name
        delegations = self._find_kwargs_delegations(func_def, kwarg_name)

        # Resolve delegation targets to actual callables
        callables: list[Callable[..., Any]] = []
        for callee_info in delegations:
            resolved = self._resolve_callee(callee_info, func, context_class)
            if resolved is not None:
                callables.append(resolved)

        return callables

    def _find_function_def(self, tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """Find a function definition by name in an AST.

        Args:
            tree: The AST module to search.
            name: The function name to find.

        Returns:
            The function definition node, or None if not found.
        """
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return node
        return None

    def _get_kwarg_param_name(self, func_def: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
        """Get the name of the **kwargs parameter.

        Args:
            func_def: The function definition AST node.

        Returns:
            The kwargs parameter name (e.g., 'kwargs'), or None if no **kwargs.
        """
        if func_def.args.kwarg:
            return func_def.args.kwarg.arg
        return None

    def _find_kwargs_delegations(
        self,
        func_def: ast.FunctionDef | ast.AsyncFunctionDef,
        kwarg_name: str,
    ) -> list[dict[str, Any]]:
        """Find all calls that receive **kwargs delegation.

        Args:
            func_def: The function definition AST node.
            kwarg_name: The name of the **kwargs parameter.

        Returns:
            List of callee info dicts containing 'type' and relevant identifiers.
        """
        delegations: list[dict[str, Any]] = []

        for node in ast.walk(func_def):
            if not isinstance(node, ast.Call):
                continue

            # Check if any keyword argument is **kwarg_name
            has_kwargs_spread = False
            for keyword in node.keywords:
                # keyword.arg is None for **something unpacking
                if keyword.arg is None and isinstance(keyword.value, ast.Name):
                    if keyword.value.id == kwarg_name:
                        has_kwargs_spread = True
                        break

            if not has_kwargs_spread:
                continue

            # Extract callee information
            callee_info = self._extract_callee_info(node.func)
            if callee_info:
                delegations.append(callee_info)

        return delegations

    def _extract_callee_info(self, node: ast.expr) -> dict[str, Any] | None:
        """Extract information about the callee from a Call.func node.

        Args:
            node: The AST expression representing the callee.

        Returns:
            Dictionary with callee information, or None if cannot extract.
        """
        # Simple name: SomeClass(**kwargs) or some_func(**kwargs)
        if isinstance(node, ast.Name):
            return {"type": "name", "name": node.id}

        # Attribute access: self.method(**kwargs), module.func(**kwargs)
        if isinstance(node, ast.Attribute):
            # Check for super().__init__(**kwargs) pattern
            if isinstance(node.value, ast.Call):
                inner_func = node.value.func
                if isinstance(inner_func, ast.Name) and inner_func.id == "super":
                    return {"type": "super", "method": node.attr}

            # Check for self.method(**kwargs)
            if isinstance(node.value, ast.Name):
                if node.value.id == "self":
                    return {"type": "self_method", "method": node.attr}

                # Could be ClassName.method(self, **kwargs) or module.func(**kwargs)
                return {"type": "attribute", "object": node.value.id, "attr": node.attr}

        return None

    def _resolve_callee(
        self,
        callee_info: dict[str, Any],
        func: Callable[..., Any],
        context_class: type | None,
    ) -> Callable[..., Any] | None:
        """Resolve callee info to an actual callable.

        Args:
            callee_info: Dictionary with callee information.
            func: The function being analyzed (for scope resolution).
            context_class: The class containing the function, if any.

        Returns:
            The resolved callable, or None if cannot resolve.
        """
        callee_type = callee_info["type"]

        # Handle super().__init__ or super().method()
        if callee_type == "super":
            if context_class is None:
                return None

            method_name = callee_info["method"]
            # Get the parent class from MRO
            mro = context_class.__mro__
            if len(mro) < 2:
                return None

            parent_class = mro[1]

            # If method is __init__, return the parent class itself
            if method_name == "__init__":
                return parent_class

            # Otherwise, get the method from parent
            if hasattr(parent_class, method_name):
                return getattr(parent_class, method_name)

            return None

        # Handle simple names (look up in function's global/local scope)
        if callee_type == "name":
            name = callee_info["name"]
            # Check function's globals
            if hasattr(func, "__globals__") and name in func.__globals__:
                return func.__globals__[name]
            return None

        # Handle self.method - look up method in context_class
        if callee_type == "self_method":
            if context_class is None:
                return None

            method_name = callee_info["method"]
            if hasattr(context_class, method_name):
                return getattr(context_class, method_name)
            return None

        # Handle module.attr or ClassName.method
        if callee_type == "attribute":
            obj_name = callee_info["object"]
            attr_name = callee_info["attr"]

            # Look up the object in function's globals
            if hasattr(func, "__globals__") and obj_name in func.__globals__:
                obj = func.__globals__[obj_name]
                if hasattr(obj, attr_name):
                    return getattr(obj, attr_name)

            return None

        return None
