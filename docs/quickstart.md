# Quick Start

Get started with SynConf in just a few minutes.

## Installation

```bash
pip install synconf
```

## Basic Usage

```python
from synconf import SynConfParser

# Define your classes
class Optimizer:
    def __init__(self, lr: float, weight_decay: float = 0.0):
        """Base optimizer.
        
        Args:
            lr: Learning rate.
            weight_decay: L2 regularization.
        """
        self.lr = lr
        self.weight_decay = weight_decay

class Adam(Optimizer):
    def __init__(self, beta1: float = 0.9, **kwargs):
        """Adam optimizer."""
        super().__init__(**kwargs)
        self.beta1 = beta1

# Create parser and register types
parser = SynConfParser()
parser.add("optimizer", Adam)

# Parse config from CLI or YAML
config = parser.parse("--optimizer.lr=0.001")

# Initialize object
opt = config.optimizer.init()  # or config.optimizer.call()
```

## Using YAML Files

Create a `config.yaml`:

```yaml
optimizer:
  TYPE: Adam
  lr: 0.001
  beta1: 0.9
  weight_decay: 0.0001
```

Parse it:

```python
config = parser.parse("config.yaml")
optimizer = config.optimizer.init()
```

## Mixing YAML and CLI

CLI arguments override YAML values:

```python
# CLI overrides take precedence
config = parser.parse("config.yaml", "--optimizer.lr=0.01")
```

Multiple sources are merged left-to-right (later overrides earlier):

```python
config = parser.parse(
    "base.yaml",           # Base configuration
    "experiment.yaml",     # Experiment overrides
    "--optimizer.lr=0.01"  # CLI overrides
)
```

## Getting Help

SynConf auto-generates help messages from your code:

```python
print(parser.help())
```

Output:
```
--optimizer.TYPE=Adam               <Adam> Adam optimizer.
--optimizer.beta1=0.9               <float> First moment decay. [default=0.9, from=Adam]
--optimizer.lr=???                  <float> Learning rate. [required, from=Optimizer]
--optimizer.weight_decay=0.0        <float> L2 regularization. [default=0.0, from=Optimizer]
```

Use `--help` flag in CLI:

```bash
python your_script.py --help
```

## Next Steps

- Learn about [Core Concepts](core-concepts.md) - initialization, TYPE system, help generation
- Master [Interpolation](interpolation.md) - dynamic value references and expressions
- Explore [Advanced Features](advanced.md) - REMOVE, LIST, and sweep functionality
- Understand [Validation](validation.md) - type checking and error handling
