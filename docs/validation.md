# Validation

SynConf validates configurations at multiple stages to catch errors early and provide clear error messages.

## Type Hints Validation

Arguments are validated against signature type hints using beartype for runtime type checking:

```python
class Model:
    def __init__(self, hidden_size: int, lr: float):
        pass

parser = SynConfParser()
parser.add("model", Model)

# Valid
config = parser.parse("--model.hidden_size=128", "--model.lr=0.01")

# Invalid - raises validation error
config = parser.parse("--model.hidden_size=not_an_int")
# ValueError: Configuration validation failed:
#   Invalid type for 'model.hidden_size': expected int, got str ('not_an_int')
```

## Strict Validation

By default, SynConf performs strict validation checking for:

1. **Missing required parameters** - Parameters without defaults that aren't provided
2. **Unknown parameters** - Parameters provided that don't exist in the spec
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
#   Missing required parameter: 'model.hidden_size'
```

### Unknown Parameters

```python
class Model:
    def __init__(self, hidden_size: int):
        pass

parser = SynConfParser()
parser.add("model", Model)

# 'unknown_param' doesn't exist in Model
config = parser.parse("--model.hidden_size=128", "--model.unknown_param=value")
# ValueError: Configuration validation failed:
#   Unknown parameter: 'model.unknown_param'
```

This catches typos and configuration errors early:

```bash
# Typo: 'hidden_siez' instead of 'hidden_size'
python train.py --model.hidden_siez=128
# Error: Unknown parameter: 'model.hidden_siez'
```

## TYPE Compatibility

### Top-Level TYPE Compatibility

`TYPE` values are checked against the registered base class:

```python
class Optimizer:
    def __init__(self, lr: float):
        pass

class Adam(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Scheduler:  # Not related to Optimizer
    pass

parser = SynConfParser()
parser.add("optimizer", Optimizer)
parser._resolver.register("Adam", Adam)
parser._resolver.register("Scheduler", Scheduler)

# Valid - Adam is subclass of Optimizer
config = parser.parse("--optimizer.TYPE=Adam", "--optimizer.lr=0.001")

# Invalid - Scheduler is not subclass of Optimizer
config = parser.parse("--optimizer.TYPE=Scheduler")
# TypeError: TYPE Scheduler is not a subclass of expected type Optimizer
```

### Nested TYPE Compatibility

When a parameter has a type hint and receives a nested config with `TYPE`, SynConf validates compatibility.

**For classes:** Checks if TYPE is a subclass of the parameter's type hint.

```python
class Optimizer:
    def __init__(self, lr: float = 0.001):
        self.lr = lr

class Adam(Optimizer):
    def __init__(self, beta1: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.beta1 = beta1

class Model:
    def __init__(self, hidden_size: int, optimizer: Optimizer):
        """Model with an optimizer.
        
        Args:
            hidden_size: Hidden layer size.
            optimizer: Optimizer instance.
        """
        self.hidden_size = hidden_size
        self.optimizer = optimizer

parser = SynConfParser()
parser.add("model", Model)
parser._resolver.register("Adam", Adam)
```

In your YAML config:

```yaml
model:
  TYPE: Model
  hidden_size: 64
  optimizer:
    TYPE: Adam  # ✓ Valid - Adam is subclass of Optimizer
    lr: 0.01
```

If you use an incompatible type:

```yaml
model:
  TYPE: Model
  hidden_size: 64
  optimizer:
    TYPE: Scheduler  # ✗ Invalid - Scheduler is NOT a subclass of Optimizer
```

This will raise a `TypeError` during `parse()`:
```
TypeError: TYPE Scheduler is not a subclass of expected type Optimizer
```

**For functions:** Checks if the return type annotation matches the parameter's type hint.

```python
def build_optimizer(opt_type: str = "adam") -> Optimizer:
    """Factory function that returns an Optimizer.
    
    Args:
        opt_type: Type of optimizer to build.
        
    Returns:
        An Optimizer instance.
    """
    if opt_type == "adam":
        return Adam()
    return Optimizer()

parser._resolver.register("build_optimizer", build_optimizer)
```

```yaml
model:
  TYPE: Model
  hidden_size: 64
  optimizer:
    TYPE: build_optimizer  # ✓ Valid - returns Optimizer
    opt_type: adam
```

### Multi-Level Nesting

Compatibility checking works recursively through arbitrary nesting levels:

```yaml
model:
  TYPE: Model
  optimizer:
    TYPE: Adam
    scheduler:
      TYPE: StepLR  # Validated against scheduler type hint in Adam
      step_size: 10
```

## Error Messages

SynConf provides clear, actionable error messages:

### Type Mismatch
```
Invalid type for 'model.hidden_size': expected int, got str ('not_an_int')
```

### Missing Required
```
Missing required parameter: 'model.hidden_size'
```

### Unknown Parameter
```
Unknown parameter: 'model.unknown_param'
```

### TYPE Incompatibility
```
TypeError: TYPE Scheduler is not a subclass of expected type Optimizer
```

### Circular Reference
```
Circular reference detected: a -> b -> c -> a
```

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
