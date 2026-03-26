# Validation

SynConf validates configurations at multiple stages to catch errors early and provide clear error messages.

## Type Hints Validation

Arguments are validated against signature type hints. All errors are **collected at once** and reported together in a single `ValueError`:

```python
class Model:
    def __init__(self, hidden_size: int, lr: float):
        pass

parser = SynConfParser()
parser.add("model", Model)

# Raises ValueError listing all errors at once:
config = parser.parse(
    "--model.hidden_size=not_an_int",
    "--model.lr=also_bad",
    "--model.unknown_param=value",
)
# ValueError: Configuration validation failed:
#   Invalid type for 'model.hidden_size': expected int, got str 'not_an_int' [source: CLI, owner: Model]
#   Invalid type for 'model.lr': expected float, got str 'also_bad' [source: CLI, owner: Model]
#   Unexpected parameter: 'model.unknown_param' [source: CLI, owner: Model]
```

Each error line includes:
- **`source`**: where the invalid value came from (`CLI`, `config.yaml`, `default value`, or `manual assignment`)
- **`owner`**: the class or function whose signature defines the parameter

## Strict Validation

SynConf validates three categories of error, all collected and reported together:

1. **Missing required parameters** - Parameters without defaults that aren't provided
2. **Unexpected parameters** - Parameters provided that don't exist in the spec
3. **Type mismatches** - Values that don't match type hints

### Missing Required Parameters

```python
class Model:
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        pass

parser = SynConfParser()
parser.add("model", Model)

# Missing required parameter 'hidden_size'
config = parser.parse("--model.dropout=0.2")
# ValueError: Configuration validation failed:
#   Missing required parameter: 'model.hidden_size' [owner: Model]
```

### Unexpected Parameters

```python
class Model:
    def __init__(self, hidden_size: int):
        pass

parser = SynConfParser()
parser.add("model", Model)

# 'unknown_param' doesn't exist in Model
config = parser.parse("--model.hidden_size=128", "--model.unknown_param=value")
# ValueError: Configuration validation failed:
#   Unexpected parameter: 'model.unknown_param' [source: CLI, owner: Model]
```

This catches typos and configuration errors early:

```bash
# Typo: 'hidden_siez' instead of 'hidden_size'
python train.py --model.hidden_siez=128
# Error: Unexpected parameter: 'model.hidden_siez' [source: CLI, owner: Model]
```

### Standalone Arguments

Arguments registered without a callable (via `type_hint=`) omit the `owner` tag:

```python
parser.add("num_trials", type_hint=int)
parser.parse()  # missing value
# ValueError: Configuration validation failed:
#   Missing required parameter: 'num_trials'
```

## TYPE Compatibility

`TYPE` values are checked against the registered base class. An incompatible `TYPE` raises `ValueError` with the standard validation error format:

```python
class Optimizer:
    def __init__(self, lr: float):
        pass

class Adam(Optimizer):
    pass

class Scheduler:  # Not related to Optimizer
    pass

parser = SynConfParser()
parser.add("optimizer", Optimizer)

# Valid - Adam is subclass of Optimizer
config = parser.parse("--optimizer.TYPE=my_module.Adam", "--optimizer.lr=0.001")

# Invalid - Scheduler is not subclass of Optimizer
config = parser.parse("--optimizer.TYPE=my_module.Scheduler")
# ValueError: Configuration validation failed:
#   TYPE Scheduler is not a subclass of expected type Optimizer [source: CLI]
```

The source tag shows where the incompatible TYPE was specified (`CLI` or a filename).

## Validation on Assignment

After parsing, assigning an invalid value to a config field raises `TypeError` immediately:

```python
parser = SynConfParser()
parser.add("model", Model)
config = parser.parse("--model.lr=0.1")

# Raises TypeError right away
config.model.lr = "bad_string"
# TypeError: Invalid type for 'lr': expected float, got str 'bad_string' [source: manual assignment, owner: Model]

# Valid assignment works
config.model.lr = 0.05
```

## Error Messages

SynConf provides clear, actionable error messages. Each error includes where the value came from (`source`) and which class or function defines the parameter (`owner`):

```
Configuration validation failed:
  Invalid type for 'model.lr': expected float, got str 'not_a_float' [source: CLI, owner: Model]
  Unexpected parameter: 'model.unknown_param' [source: CLI, owner: Model]
  Missing required parameter: 'trainer.max_epochs' [owner: Trainer]
  Invalid type for 'optimizer_fn.lr': expected float, got str 'not_a_float' [source: config.yaml, owner: create_optimizer]
  Invalid type for 'seed': expected int, got str 'not_an_int' [source: default value, owner: _bad_seed]
  Unexpected parameter: 'nonexistent' [source: CLI]
  Missing required parameter: 'num_trials'
```

Source values:
- `CLI` — value came from a command-line argument
- `config.yaml` — value came from a YAML file (filename shown)
- `default value` — the parameter's own default value has a wrong type
- `manual assignment` — value was assigned directly after parsing

## Validation Order

Validation happens after all resolution steps:

1. Load sources
2. Merge configurations
3. Resolve TYPE keys (including LIST)
4. Impute default values
5. Resolve interpolations
6. **Validate** ← Catches all errors here

This ensures that:
- Interpolated values are validated
- Default values are included in validation
- TYPE-specific parameters are checked correctly

## Best Practices

### Use Type Hints Everywhere

```python
# ✓ Good - clear type hints
def train(lr: float, epochs: int, debug: bool = False):
    pass

# ✗ Avoid - no type hints, can't validate
def train(lr, epochs, debug=False):
    pass
```

### Provide Descriptive Docstrings

```python
class Optimizer:
    def __init__(self, lr: float, weight_decay: float = 0.0):
        """Base optimizer.
        
        Args:
            lr: Learning rate for parameter updates.
            weight_decay: L2 regularization coefficient.
        """
```

Docstrings appear in help messages and make errors clearer.

### Test Configuration Early

```python
# Validate config before long training runs
try:
    config = parser.parse_args()
except ValueError as e:
    print(f"Configuration error: {e}")
    sys.exit(1)

# All validation passed - safe to proceed
model = config.model.init()
```

## Next Steps

- Master [Interpolation](interpolation.md) for dynamic values and expressions
- Explore [Advanced Features](advanced.md) for REMOVE, LIST, and sweep functionality
- Review [Core Concepts](core-concepts.md) for TYPE system and help generation
