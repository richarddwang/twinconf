"""SynConfParser - Main parser for configuration management."""

from __future__ import annotations

import copy
import itertools
from typing import Any, Callable

import yaml

from synconf.config import Config
from synconf.help import HelpBuilder
from synconf.introspection import CallableSpec, Introspector
from synconf.loading.cli_loader import CliLoader, is_cli_arg
from synconf.loading.merger import Merger
from synconf.loading.yaml_loader import YamlLoader, is_yaml_file
from synconf.resolution.interpolator import Interpolator
from synconf.resolution.resolver import TYPE_KEY, Resolver
from synconf.validation import Validator


class SynConfParser:
    """Parser for configuration management synced with code.

    Provides:
    - Auto-schema detection from function/class signatures
    - **kwargs tracing through call chains
    - YAML + CLI source merging in order
    - TYPE resolution and compatibility checking
    - Type validation
    - Help generation showing parameter sources

    Example:
        >>> parser = SynConfParser()
        >>> parser.add("optimizer", Adam)
        >>> config = parser.parse("config.yaml", "--optimizer.lr=0.01")
        >>> optimizer = config.optimizer.instantiate()
    """

    def __init__(self) -> None:
        """Initialize the parser."""
        self._introspector = Introspector()
        self._yaml_loader = YamlLoader()
        self._cli_loader = CliLoader()
        self._merger = Merger()
        self._interpolator = Interpolator()
        self._resolver = Resolver()
        self._validator = Validator()
        self._help_builder = HelpBuilder(self._introspector, self._resolver)

        # Registered groups: group_key -> (CallableSpec, expected_type)
        self._groups: dict[str, tuple[CallableSpec, type | None]] = {}

    def add(self, group_key: str, target: type | Callable[..., Any]) -> None:
        """Register a callable under a group key.

        Args:
            group_key: The key under which this callable's parameters will be grouped.
            target: A class, function, or method to introspect.

        Example:
            >>> parser.add("optimizer", Adam)
            >>> # Config will have structure: optimizer: { TYPE: Adam, lr: ..., ... }
        """
        # Introspect the target
        spec = self._introspector.analyze(target)

        # Determine expected type for compatibility checking
        expected_type: type | None = None
        if isinstance(target, type):
            expected_type = target

        # Register with resolver
        self._resolver.register(spec.name, target)

        # Also register by simple class/function name for convenience
        if hasattr(target, "__name__"):
            self._resolver.register(target.__name__, target)

        # Register all types found through kwargs tracing
        self._register_delegees(spec)

        # Store the group
        self._groups[group_key] = (spec, expected_type)

    def parse(self, *sources: str) -> Config:
        """Parse configuration from multiple sources.

        Sources are processed in order. Later sources override earlier ones.
        Sources can be YAML file paths or CLI arguments (starting with --).

        Resolution order:
        1. Load sources
        2. Merge configurations
        3. Resolve TYPE keys (including LIST conversion)
        4. Impute default values from specs
        5. Resolve interpolations (~{...})
        6. Validate against type hints

        Args:
            *sources: YAML file paths or CLI arguments.

        Returns:
            Config object with merged and validated configuration.

        Example:
            >>> config = parser.parse("base.yaml", "--lr=0.01", "override.yaml")
        """
        # 1. Load all sources
        configs: list[dict[str, Any]] = []

        for source in sources:
            if is_yaml_file(source):
                configs.append(self._yaml_loader.load(source))
            elif is_cli_arg(source):
                configs.append(self._cli_loader.parse(source))
            else:
                raise ValueError(f"Unknown source type: {source} (must be .yaml/.yml file or --arg=value)")

        # 2. Merge in order
        merged = self._merger.merge_all(configs)

        # 3-6. Resolve, impute, interpolate, validate
        return self._resolve_merged(merged)

    def parse_args(self, args: list[str] | None = None) -> Config | list[Config]:
        """Parse configuration from command line arguments.

        Convenience method that handles sys.argv parsing and sweep subcommand.

        Args:
            args: List of arguments (defaults to sys.argv[1:]).

        Returns:
            Config object if no sweep, or list of Config objects if sweep is used.
        """
        import sys

        if args is None:
            args = sys.argv[1:]

        # Check for --help or --help.<key_path>
        key_path_filter = None
        for arg in args:
            if arg == "--help" or arg == "-h":
                print(self.help())
                sys.exit(0)
            elif arg.startswith("--help."):
                key_path_filter = arg[7:]  # Extract key path after "--help."
                print(self.help(key_path_filter=key_path_filter))
                sys.exit(0)

        # Check for sweep subcommand
        base_args, sweep_values = self._parse_sweep_args(args)

        if not sweep_values:
            # No sweep - normal parsing
            return self.parse(*base_args)

        # Sweep mode - generate all combinations
        # First, load and merge base configuration
        configs: list[dict[str, Any]] = []

        for source in base_args:
            if is_yaml_file(source):
                configs.append(self._yaml_loader.load(source))
            elif is_cli_arg(source):
                configs.append(self._cli_loader.parse(source))
            else:
                raise ValueError(f"Unknown source type: {source} (must be .yaml/.yml file or --arg=value)")

        base_merged = self._merger.merge_all(configs)

        # Generate all sweep combinations
        sweep_configs = self._generate_sweep_configs(base_merged, sweep_values)

        # Resolve each swept config using shared resolution logic
        return [self._resolve_merged(sweep_config) for sweep_config in sweep_configs]

    def help(self, config_data: dict[str, Any] | None = None, key_path_filter: str | None = None) -> str:
        """Generate help text for all registered groups.

        Args:
            config_data: Optional current configuration values to display.
            key_path_filter: Optional key path to filter arguments (e.g., "optimizer" or "trainer.optimizer").

        Returns:
            Formatted help text.
        """
        specs = {key: group[0] for key, group in self._groups.items()}
        return self._help_builder.build(specs, config_data, key_path_filter)

    def get_spec(self, group_key: str) -> CallableSpec | None:
        """Get the CallableSpec for a group.

        Args:
            group_key: The group key.

        Returns:
            The CallableSpec, or None if not found.
        """
        if group_key in self._groups:
            return self._groups[group_key][0]
        return None

    def _resolve_merged(self, merged: dict[str, Any]) -> Config:
        """Apply resolution pipeline to merged config.

        Resolution steps:
        1. Resolve TYPE keys (including LIST conversion)
        2. Impute default values from specs
        3. Resolve ~{...} interpolations
        4. Validate against type hints

        Args:
            merged: Merged configuration dictionary.

        Returns:
            Validated Config object.
        """
        # Resolve TYPE keys for each group (with its spec for nested validation)
        resolved: dict[str, Any] = {}
        for group_key, (spec, expected_type) in self._groups.items():
            if group_key in merged:
                # Build expected_types for backward compatibility (top-level only)
                expected_types: dict[str, type] = {}
                if expected_type is not None:
                    expected_types[group_key] = expected_type

                # Resolve this group with its spec for nested TYPE validation
                resolved[group_key] = self._resolver.resolve_in_config(
                    {group_key: merged[group_key]}, expected_types, spec=spec
                )[group_key]

        # Also resolve any top-level keys not in groups
        for key in merged:
            if key not in self._groups and key not in resolved:
                resolved[key] = merged[key]

        # Impute default values from specs
        for group_key, (spec, _) in self._groups.items():
            if group_key in resolved:
                # Inject default TYPE if missing
                if TYPE_KEY not in resolved[group_key]:
                    resolved[group_key][TYPE_KEY] = spec.callable_obj

                # Impute other default parameter values
                self._impute_defaults(resolved[group_key], spec)

        # Resolve ~{...} interpolations
        resolved = self._interpolator.resolve_all(resolved)

        # Validate configuration
        for group_key, (spec, _) in self._groups.items():
            if group_key in resolved:
                self._validator.validate_or_raise(resolved[group_key], spec, group_key)

        return Config(resolved)

    def _parse_sweep_args(self, args: list[str]) -> tuple[list[str], dict[str, list[Any]]]:
        """Parse sweep arguments from command line.

        Args:
            args: Command line arguments.

        Returns:
            Tuple of (base_args, sweep_values) where sweep_values maps key paths to value lists.
        """
        # Find "sweep" keyword position
        if "sweep" not in args:
            return args, {}

        sweep_idx = args.index("sweep")
        base_args = args[:sweep_idx]
        sweep_args = args[sweep_idx + 1 :]

        # Parse sweep arguments (format: --key=[val1,val2,...])
        sweep_values: dict[str, list[Any]] = {}

        for arg in sweep_args:
            if not arg.startswith("--"):
                raise ValueError(f"Sweep argument must start with --: {arg}")

            if "=" not in arg:
                raise ValueError(f"Sweep argument must have format --key=[val1,val2,...]: {arg}")

            key_path, value_str = arg[2:].split("=", 1)

            # Parse list value (should be YAML list format like [1,2,3])
            try:
                values = yaml.safe_load(value_str)
                if not isinstance(values, list):
                    raise ValueError(f"Sweep value must be a list: {arg}")
                sweep_values[key_path] = values
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML list in sweep argument {arg}: {e}")

        return base_args, sweep_values

    def _generate_sweep_configs(
        self, base_config: dict[str, Any], sweep_values: dict[str, list[Any]]
    ) -> list[dict[str, Any]]:
        """Generate all sweep combinations from base config and sweep values.

        Args:
            base_config: Base configuration dictionary.
            sweep_values: Mapping of key paths to value lists.

        Returns:
            List of configuration dictionaries with all sweep combinations.
        """
        if not sweep_values:
            return [base_config]

        # Get key paths and their corresponding value lists (in order)
        key_paths = list(sweep_values.keys())
        value_lists = [sweep_values[key] for key in key_paths]

        # Generate all combinations using itertools.product
        configs = []
        for combo in itertools.product(*value_lists):
            # Deep copy base config for this combination
            config = copy.deepcopy(base_config)

            # Apply sweep values for this combination
            for key_path, value in zip(key_paths, combo):
                self._set_nested_value(config, key_path, value)

            configs.append(config)

        return configs

    def _set_nested_value(self, config: dict[str, Any], key_path: str, value: Any) -> None:
        """Set a value in a nested dictionary using dotted key path.

        Args:
            config: Configuration dictionary to modify.
            key_path: Dotted key path like "model.lr".
            value: Value to set.
        """
        parts = key_path.split(".")
        current = config

        # Navigate to parent
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                raise ValueError(f"Cannot set nested value: {part} in {key_path} is not a dict")
            current = current[part]

        # Set final value
        current[parts[-1]] = value

    def _impute_defaults(self, config: dict[str, Any], spec: CallableSpec) -> None:
        """Impute default values from spec into config.

        Args:
            config: Configuration dictionary to modify.
            spec: CallableSpec with parameter defaults.
        """
        all_params = spec.get_all_params()

        for param_name, param_spec in all_params.items():
            # Skip if already in config
            if param_name in config:
                continue

            # Skip if no default available
            if param_spec.is_required:
                continue

            # Impute the default value
            config[param_name] = param_spec.default

    def _register_delegees(self, spec: CallableSpec) -> None:
        """Register all callables found through kwargs delegation.

        Args:
            spec: The CallableSpec to process.
        """
        for delegee_spec in spec.kwargs_delegees:
            # Register the delegee callable
            self._resolver.register(delegee_spec.name, delegee_spec.callable_obj)
            if hasattr(delegee_spec.callable_obj, "__name__"):
                self._resolver.register(delegee_spec.callable_obj.__name__, delegee_spec.callable_obj)

            # Recursively register delegees of delegees
            self._register_delegees(delegee_spec)
