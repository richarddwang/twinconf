# SynConf Documentation

SynConf is a configuration management system that syncs with your code. It auto-detects argument schemas from function/class signatures and traces `**kwargs` through call chains.

## Documentation Overview

- **[Quick Start](docs/quickstart.md)** - Get started with SynConf in 5 minutes
- **[Core Concepts](docs/core-concepts.md)** - Learn about initialization, TYPE system, help generation, and configuration sources
- **[Validation](docs/validation.md)** - Understand type hints validation, TYPE compatibility checking, and strict validation
- **[Interpolation](docs/interpolation.md)** - Master variable references and expression evaluation with `~{...}` syntax
- **[Advanced Features](docs/advanced.md)** - Explore REMOVE, LIST, and sweep functionality

## Key Features

### Auto-Schema Detection
SynConf introspects function and class signatures to automatically extract:
- Parameter names and types
- Default values
- Docstring descriptions
- `**kwargs` delegation chains

### Multiple Configuration Sources
Merge YAML files and CLI arguments in any order:
```python
config = parser.parse("base.yaml", "--optimizer.lr=0.01", "override.yaml")
```

### Type Safety
Automatic validation against type hints with clear error messages:
```python
class Model:
    def __init__(self, hidden_size: int): ...

# This will raise a validation error:
config = parser.parse("--model.hidden_size=not_an_int")
```

### Dynamic Interpolation
Reference values and evaluate expressions with `~{...}` syntax:
```yaml
base_lr: 0.001
optimizer:
  lr: ~{base_lr * 10}  # Expression: 0.01
```

### Parameter Sweeps
Run multiple configurations with Cartesian product sweeps:
```bash
python train.py config.yaml sweep --model.lr=[1e-4,2e-5] --seed=[0,1,2]
# Generates 6 configurations (2 × 3)
```

## Installation

```bash
pip install synconf
```

## Quick Example

```python
from synconf import SynConfParser

class Optimizer:
    def __init__(self, lr: float, weight_decay: float = 0.0):
        """Base optimizer."""
        self.lr = lr
        self.weight_decay = weight_decay

# Create parser and register
parser = SynConfParser()
parser.add("optimizer", Optimizer)

# Parse from YAML or CLI
config = parser.parse("config.yaml", "--optimizer.lr=0.001")

# Instantiate object
optimizer = config.optimizer.init()
```

## License

MIT License - see LICENSE file for details.
