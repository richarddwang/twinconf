"""Config class for accessing and instantiating configuration."""

from __future__ import annotations

from typing import Any, Iterator

from beartype.door import is_bearable

from synconf.introspection.spec import CallableSpec
from synconf.resolution.resolver import TYPE_KEY


class Config:
    """Configuration wrapper with dict-style and attribute-style access.

    Provides:
    - Dual access: config['key'] and config.key both work
    - Sub-configs: Nested dicts become Config objects
    - Instantiation: config.instantiate() calls the TYPE callable with config values

    Attributes:
        _data: The underlying configuration dictionary.
        _spec: Optional CallableSpec for validation and introspection.
    """

    def __init__(
        self,
        data: dict[str, Any],
        spec: CallableSpec | None = None,
        path: str = "",
    ) -> None:
        """Initialize a Config from a dictionary.

        Args:
            data: The configuration dictionary.
            spec: Optional CallableSpec for this config level.
            path: Dot-separated path identifying this config's position in the tree.
        """
        # Use object.__setattr__ to avoid triggering __setattr__
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_path", path)

    def __getitem__(self, key: str) -> Any:
        """Get a value by key (dict-style access).

        Supports dot-notation for nested access (e.g. ``config["optimizer.lr"]``).
        Direct key lookup takes priority; dot-splitting is only attempted when
        the literal key is absent.

        Args:
            key: The configuration key, optionally dot-separated for nested access.

        Returns:
            The value, wrapped in Config if it's a dict.

        Raises:
            KeyError: If the key (or any segment of a dotted path) doesn't exist.
        """
        # Direct key lookup takes priority over dot-notation splitting
        if key in self._data:
            return self._wrap_value(self._data[key], key)

        # Attempt dot-notation traversal
        if "." in key:
            parts = key.split(".")
            current: Any = self
            for part in parts:
                if not isinstance(current, Config):
                    raise KeyError(f"Config key not found: '{key}'")
                current = current[part]
            return current

        raise KeyError(f"Config key not found: '{key}'")

    def __getattr__(self, name: str) -> Any:
        """Get a value by attribute name (attribute-style access).

        Args:
            name: The attribute/key name.

        Returns:
            The value, wrapped in Config if it's a dict with TYPE.

        Raises:
            AttributeError: If the attribute doesn't exist.
        """
        # Avoid recursion for special attributes
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        try:
            return self.__getitem__(name)
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a value by key, validating against spec if available.

        Supports dot-notation for nested assignment (e.g. ``config["optimizer.lr"] = 0.1``).
        Direct key takes priority; dot-splitting is only attempted when the literal key
        is absent and the key contains a dot.

        Args:
            key: The configuration key, optionally dot-separated for nested access.
            value: The value to set.

        Raises:
            TypeError: If the value doesn't match the spec's type hint.
            KeyError: If an intermediate segment of a dotted path doesn't exist.
        """
        # Direct key assignment (including literal dotted keys already in _data)
        if "." not in key or key in self._data:
            self._validate_assignment(key, value)
            self._data[key] = value
            return

        # Dot-notation traversal to the parent, then set the leaf
        parts = key.split(".")
        current: Any = self
        for part in parts[:-1]:
            if not isinstance(current, Config):
                raise KeyError(f"Cannot set '{key}': intermediate key '{part}' is not a Config")
            current = current[part]
        if not isinstance(current, Config):
            raise KeyError(f"Cannot set '{key}': parent is not a Config")
        current[parts[-1]] = value

    def __setattr__(self, name: str, value: Any) -> None:
        """Set a value by attribute name, validating against spec if available.

        Args:
            name: The attribute/key name.
            value: The value to set.

        Raises:
            TypeError: If the value doesn't match the spec's type hint.
        """
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._validate_assignment(name, value)
            self._data[name] = value

    def __contains__(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: The key to check.

        Returns:
            True if the key exists.
        """
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys.

        Returns:
            Iterator of keys.
        """
        return iter(self._data)

    def __len__(self) -> int:
        """Get the number of keys.

        Returns:
            Number of keys.
        """
        return len(self._data)

    def __repr__(self) -> str:
        """Get string representation.

        Returns:
            String representation.
        """
        type_str = ""
        if TYPE_KEY in self._data:
            type_val = self._data[TYPE_KEY]
            if callable(type_val):
                type_str = f" TYPE={getattr(type_val, '__name__', str(type_val))}"
            else:
                type_str = f" TYPE={type_val}"

        return f"Config({dict(self._data)!r}{type_str})"

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with a default.

        Args:
            key: The configuration key.
            default: Default value if key doesn't exist.

        Returns:
            The value or default.
        """
        if key in self._data:
            return self._wrap_value(self._data[key], key)
        return default

    def keys(self) -> list[str]:
        """Get all keys.

        Returns:
            List of keys.
        """
        return list(self._data.keys())

    def values(self) -> list[Any]:
        """Get all values.

        Returns:
            List of values (with dicts wrapped as Config).
        """
        return [self._wrap_value(v, k) for k, v in self._data.items()]

    def items(self) -> list[tuple[str, Any]]:
        """Get all key-value pairs.

        Returns:
            List of (key, value) tuples (with dicts wrapped as Config).
        """
        return [(k, self._wrap_value(v, k)) for k, v in self._data.items()]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary.

        Returns:
            Plain dictionary (nested Configs also converted).
        """
        result: dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, dict):
                result[key] = self._dict_to_plain(value)
            else:
                result[key] = value
        return result

    def init(self, **overwrites: Any) -> Any:
        """Initialize the object or call the function defined by TYPE.

        This method replaces the previous `instantiate()` method.

        Args:
            **overwrites: Keyword arguments to override config values.

        Returns:
            The initialized object or function result.

        Raises:
            ValueError: If TYPE is missing or not callable.
        """
        # Include path in error messages for clear diagnostics
        path_desc = f" '{self._path}'" if self._path else ""

        if TYPE_KEY not in self._data:
            raise ValueError(f"Cannot init{path_desc}: no TYPE specified in config")

        type_callable = self._data[TYPE_KEY]
        if not callable(type_callable):
            raise ValueError(f"Cannot init{path_desc}: TYPE is not callable: {type_callable}")

        # Build kwargs from config (excluding TYPE)
        kwargs = {}
        for key, value in self._data.items():
            if key == TYPE_KEY:
                continue

            child_path = f"{self._path}.{key}" if self._path else key

            # Recursively init nested configs
            if isinstance(value, dict) and TYPE_KEY in value:
                nested_config = Config(value, path=child_path)
                kwargs[key] = nested_config.init()
            # Handle lists containing configs with TYPE
            elif isinstance(value, list):
                kwargs[key] = [
                    Config(item, path=f"{child_path}[{i}]").init()
                    if isinstance(item, dict) and TYPE_KEY in item
                    else item
                    for i, item in enumerate(value)
                ]
            else:
                kwargs[key] = value

        # Apply overwrites
        kwargs.update(overwrites)

        return type_callable(**kwargs)

    def call(self, **overwrites: Any) -> Any:
        """Alias for init()."""
        return self.init(**overwrites)

    def instantiate(self, **overwrites: Any) -> Any:
        """Deprecated alias for init()."""
        return self.init(**overwrites)

    def _wrap_value(self, value: Any, key: str = "") -> Any:
        """Wrap a dict value as Config if it has a TYPE key.

        Args:
            value: The value to potentially wrap.
            key: The key used to access this value, for path propagation.

        Returns:
            Config-wrapped value if applicable, otherwise the original value.
        """
        # If value is already a Config, return as-is
        if isinstance(value, Config):
            return value

        child_path = f"{self._path}.{key}" if self._path else key

        if isinstance(value, dict) and TYPE_KEY in value:
            return Config(value, path=child_path)

        # Also wrap plain dicts for consistent access
        if isinstance(value, dict):
            return Config(value, path=child_path)

        return value

    def _validate_assignment(self, key: str, value: Any) -> None:
        """Validate a value against the spec's type hint for the given key.

        Only validates if a spec is attached and the key exists in the spec's parameters.
        New keys (not in the spec) are allowed without validation.

        Args:
            key: The configuration key being set.
            value: The value being assigned.

        Raises:
            TypeError: If the value doesn't match the spec's type hint.
        """
        if self._spec is None:
            return

        all_params = self._spec.get_all_params()
        if key not in all_params:
            return

        param = all_params[key]
        if param.type_hint is not None and not is_bearable(value, param.type_hint):
            parts = ["source: manual assignment"]
            if param.source:
                parts.append(f"owner: {param.source.name}")
            metadata = " [" + ", ".join(parts) + "]"
            raise TypeError(
                f"Invalid type for '{key}': expected {param.format_type()}, "
                f"got {type(value).__name__} {value!r}{metadata}"
            )

    def _dict_to_plain(self, d: dict[str, Any]) -> dict[str, Any]:
        """Convert a nested dict to plain dict.

        Args:
            d: Dictionary to convert.

        Returns:
            Plain dictionary.
        """
        result: dict[str, Any] = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = self._dict_to_plain(value)
            else:
                result[key] = value
        return result
