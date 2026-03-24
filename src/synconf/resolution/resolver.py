"""TYPE resolution and compatibility checking."""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any, Callable, get_type_hints

if TYPE_CHECKING:
    from synconf.introspection import CallableSpec

# Special key used to specify the callable type in config
TYPE_KEY = "TYPE"

# Special TYPE value to convert dict to list
LIST = "LIST"


class Resolver:
    """Resolves TYPE strings to actual callables and checks compatibility.

    Handles:
    - Import path strings (e.g., "torch.optim.Adam")
    - Simple names looked up in a registry
    - Compatibility checking (subclass for classes, return type for functions)
    """

    def __init__(self) -> None:
        """Initialize the Resolver with an empty registry."""
        # Registry mapping names to callables
        self._registry: dict[str, Callable[..., Any]] = {}
        # Lazy-loaded introspector for analyzing resolved callables
        self._introspector: Any = None

    def register(self, name: str, callable_obj: Callable[..., Any]) -> None:
        """Register a callable by name.

        Args:
            name: The name to register (used for TYPE lookup).
            callable_obj: The callable to register.
        """
        self._registry[name] = callable_obj

    def register_many(self, callables: dict[str, Callable[..., Any]]) -> None:
        """Register multiple callables.

        Args:
            callables: Dictionary mapping names to callables.
        """
        self._registry.update(callables)

    def resolve(self, type_value: str | type | Callable[..., Any]) -> str | Callable[..., Any]:
        """Resolve a TYPE value to an actual callable.

        Args:
            type_value: The TYPE value (string path, registered name, or callable).

        Returns:
            The resolved callable, or LIST if it's the special list marker.

        Raises:
            ValueError: If the TYPE cannot be resolved.
        """
        # Already a callable
        if callable(type_value):
            return type_value

        # Must be a string
        if not isinstance(type_value, str):
            raise ValueError(f"TYPE must be a string or callable, got {type(type_value).__name__}")

        # Handle special LIST marker
        if type_value == LIST:
            return LIST

        # Try registry first
        if type_value in self._registry:
            return self._registry[type_value]

        # Try import path
        return self._import_string(type_value)

    def resolve_in_config(
        self,
        config: dict[str, Any],
        expected_types: dict[str, type] | None = None,
        spec: CallableSpec | None = None,
    ) -> dict[str, Any]:
        """Resolve all TYPE keys in a config dictionary.

        Args:
            config: The configuration dictionary.
            expected_types: Optional mapping of config paths to expected base types
                           for compatibility checking (for backward compatibility).
            spec: Optional CallableSpec providing type hints for parameters.

        Returns:
            A new config dict with TYPE values resolved to actual callables.

        Raises:
            ValueError: If a TYPE cannot be resolved or is not compatible.
        """
        expected_types = expected_types or {}
        return self._resolve_recursive(config, "", expected_types, parent_spec=spec)

    def check_compatibility(
        self,
        resolved: Callable[..., Any],
        expected: type,
    ) -> bool:
        """Check if a resolved callable is compatible with an expected type.

        For classes: checks if resolved is a subclass of expected.
        For functions: checks if return type is compatible with expected.
        For methods: similar to functions.

        Args:
            resolved: The resolved callable.
            expected: The expected base type.

        Returns:
            True if compatible.

        Raises:
            TypeError: If not compatible.
        """
        # If resolved is a class
        if isinstance(resolved, type):
            if not issubclass(resolved, expected):
                raise TypeError(f"TYPE {resolved.__name__} is not a subclass of expected type {expected.__name__}")
            return True

        # If resolved is a function or method, check return type
        if callable(resolved):
            return_type = self._get_return_type(resolved)

            # If no return type annotation, we can't check - assume compatible
            if return_type is None:
                return True

            # If return type is the expected type or a subclass
            if isinstance(return_type, type):
                if issubclass(return_type, expected):
                    return True

                raise TypeError(
                    f"TYPE {getattr(resolved, '__name__', str(resolved))} returns {return_type.__name__}, "
                    f"which is not compatible with expected type {expected.__name__}"
                )

            # For complex return types (e.g., Union), assume compatible
            return True

        return True

    def _resolve_recursive(
        self,
        obj: Any,
        path: str,
        expected_types: dict[str, type],
        parent_spec: CallableSpec | None = None,
    ) -> Any:
        """Recursively resolve TYPE keys in a nested structure.

        Args:
            obj: The object to process.
            path: Current path in the config (for error messages and expected_types lookup).
            expected_types: Mapping of paths to expected types.
            parent_spec: Optional CallableSpec of the parent callable, used to extract
                        expected types from parameter type hints.

        Returns:
            The processed object with TYPE values resolved.
            If TYPE is LIST, returns a list instead of a dict.
        """
        if not isinstance(obj, dict):
            return obj

        result = {}
        resolved_callable = None
        is_list_type = False

        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key

            if key == TYPE_KEY:
                # Resolve the TYPE value
                resolved = self.resolve(value)

                # Check if this is a LIST type
                if resolved == LIST:
                    is_list_type = True
                    # Don't include TYPE in result for lists
                    continue

                # Check compatibility if expected type is specified (top-level)
                if path in expected_types:
                    self.check_compatibility(resolved, expected_types[path])

                # Check compatibility using parent_spec's parameter type hint (nested)
                if parent_spec is not None and path:
                    # Extract the parameter name (last component of path)
                    param_name = path.split(".")[-1]
                    all_params = parent_spec.get_all_params()

                    if param_name in all_params:
                        param_spec = all_params[param_name]
                        type_hint = param_spec.type_hint

                        # Only validate if type_hint is a concrete class type
                        if isinstance(type_hint, type):
                            self.check_compatibility(resolved, type_hint)

                result[key] = resolved
                resolved_callable = resolved

            elif isinstance(value, dict):
                # Check if this nested dict has a TYPE (will need its spec for further nesting)
                nested_spec = None
                if TYPE_KEY in value and resolved_callable is not None:
                    # Get spec for the resolved callable to validate its nested configs
                    nested_spec = self._get_spec(resolved_callable)
                elif parent_spec is not None:
                    # This nested dict belongs to a parameter - try to get spec from the param's type hint
                    param_name = key
                    all_params = parent_spec.get_all_params()
                    if param_name in all_params:
                        param_type = all_params[param_name].type_hint
                        if isinstance(param_type, type):
                            nested_spec = self._get_spec(param_type)

                # Recursively process nested dicts
                result[key] = self._resolve_recursive(value, new_path, expected_types, parent_spec=nested_spec)
            else:
                result[key] = value

        # Convert to list if TYPE: LIST
        if is_list_type:
            return list(result.values())

        return result

    def _get_spec(self, callable_obj: Callable[..., Any]) -> CallableSpec | None:
        """Get the CallableSpec for a callable by introspecting it.

        Args:
            callable_obj: The callable to introspect.

        Returns:
            CallableSpec or None if introspection fails.
        """
        # Lazy-load introspector to avoid circular import
        if self._introspector is None:
            from synconf.introspection import Introspector

            self._introspector = Introspector()

        try:
            return self._introspector.analyze(callable_obj)
        except Exception:
            return None

    def _import_string(self, import_path: str) -> Callable[..., Any]:
        """Import a callable from a dotted path string.

        Args:
            import_path: Dotted path like "package.module.ClassName" or "package.module.function".

        Returns:
            The imported callable.

        Raises:
            ValueError: If the import fails.
        """
        # Split into module path and attribute name
        if "." not in import_path:
            raise ValueError(f"Cannot resolve TYPE '{import_path}': not found in registry and not a valid import path")

        parts = import_path.rsplit(".", 1)
        module_path, attr_name = parts

        # Try to import
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            # Try one level up (for nested classes like Module.NestedClass)
            if "." in module_path:
                parent_path, parent_attr = module_path.rsplit(".", 1)
                try:
                    parent_module = importlib.import_module(parent_path)
                    parent_obj = getattr(parent_module, parent_attr)
                    if hasattr(parent_obj, attr_name):
                        return getattr(parent_obj, attr_name)
                except (ImportError, AttributeError):
                    pass

            raise ValueError(f"Cannot import module '{module_path}' for TYPE '{import_path}'")

        # Get the attribute
        if not hasattr(module, attr_name):
            raise ValueError(f"Module '{module_path}' has no attribute '{attr_name}'")

        result = getattr(module, attr_name)

        if not callable(result):
            raise ValueError(f"TYPE '{import_path}' is not callable")

        return result

    def _get_return_type(self, func: Callable[..., Any]) -> Any | None:
        """Get the return type annotation of a callable.

        Args:
            func: A function or method.

        Returns:
            The return type annotation, or None if not annotated.
        """
        try:
            hints = get_type_hints(func)
            return hints.get("return")
        except Exception:
            # Fallback to direct annotation
            try:
                sig = inspect.signature(func)
                if sig.return_annotation is not inspect.Parameter.empty:
                    return sig.return_annotation
            except (ValueError, TypeError):
                pass
            return None
