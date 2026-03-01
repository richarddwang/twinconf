"""Config class for accessing and instantiating configuration."""

from __future__ import annotations

from typing import Any, Iterator

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
    ) -> None:
        """Initialize a Config from a dictionary.

        Args:
            data: The configuration dictionary.
            spec: Optional CallableSpec for this config level.
        """
        # Use object.__setattr__ to avoid triggering __setattr__
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_spec", spec)

    def __getitem__(self, key: str) -> Any:
        """Get a value by key (dict-style access).

        Args:
            key: The configuration key.

        Returns:
            The value, wrapped in Config if it's a dict with TYPE.

        Raises:
            KeyError: If the key doesn't exist.
        """
        if key not in self._data:
            raise KeyError(f"Config key not found: '{key}'")

        value = self._data[key]
        return self._wrap_value(value)

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
        """Set a value by key.

        Args:
            key: The configuration key.
            value: The value to set.
        """
        self._data[key] = value

    def __setattr__(self, name: str, value: Any) -> None:
        """Set a value by attribute name.

        Args:
            name: The attribute/key name.
            value: The value to set.
        """
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
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
            return self._wrap_value(self._data[key])
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
        return [self._wrap_value(v) for v in self._data.values()]

    def items(self) -> list[tuple[str, Any]]:
        """Get all key-value pairs.

        Returns:
            List of (key, value) tuples (with dicts wrapped as Config).
        """
        return [(k, self._wrap_value(v)) for k, v in self._data.items()]

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
        if TYPE_KEY not in self._data:
            raise ValueError("Cannot init: no TYPE specified in config")

        type_callable = self._data[TYPE_KEY]
        if not callable(type_callable):
            raise ValueError(f"Cannot init: TYPE is not callable: {type_callable}")

        # Build kwargs from config (excluding TYPE)
        kwargs = {}
        for key, value in self._data.items():
            if key == TYPE_KEY:
                continue

            # Recursively init nested configs
            if isinstance(value, dict) and TYPE_KEY in value:
                nested_config = Config(value)
                kwargs[key] = nested_config.init()
            # Handle lists containing configs with TYPE
            elif isinstance(value, list):
                kwargs[key] = [
                    Config(item).init() if isinstance(item, dict) and TYPE_KEY in item else item for item in value
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

    def _wrap_value(self, value: Any) -> Any:
        """Wrap a dict value as Config if it has a TYPE key.

        Args:
            value: The value to potentially wrap.

        Returns:
            Config-wrapped value if applicable, otherwise the original value.
        """
        if isinstance(value, dict) and TYPE_KEY in value:
            return Config(value)

        # Also wrap plain dicts for consistent access
        if isinstance(value, dict):
            return Config(value)

        return value

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
