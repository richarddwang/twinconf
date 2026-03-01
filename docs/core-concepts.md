# Core Concepts

Learn the fundamental concepts of SynConf: initialization, TYPE system, help generation, and configuration sources.

## Initialization (`init` / `call`)

Once you have a configuration object, you can convert it back into a Python object or result using `.init()` (or its alias `.call()`).

- **Classes**: `.init()` calls the constructor `__init__`.
- **Functions**: `.call()` calls the function.

```python
# For a class config
optimizer = config.optimizer.init() 

# For a function config
result = config.train_fn.call()
```

You can also pass runtime overrides:

```python
# Override learning rate at runtime
optimizer = config.optimizer.init(lr=0.01)
```

## The `TYPE` Key

The `TYPE` key in configuration specifies which callable to use. It can be:

- A registered class name (e.g., `TYPE: Adam`)
- A full python path (e.g., `TYPE: my_module.optimizers.Adam`)
- A standard library path (e.g., `TYPE: json.JSONDecoder`)

SynConf ensures the specified `TYPE` is compatible with the expected type (subclass check).

### Example

```yaml
optimizer:
  TYPE: Adam
  lr: 0.001
  beta1: 0.9
```

```python
parser = SynConfParser()
parser.add("optimizer", Optimizer)  # Base class
parser._resolver.register("Adam", Adam)  # Register Adam

config = parser.parse("config.yaml")
opt = config.optimizer.init()  # Creates Adam instance
```

## Help Generation

SynConf generates comprehensive help messages in a CLI-style format showing parameter types, descriptions, default values, and source information.

### Basic Help

```python
print(parser.help())
```

Output format:

```
--optimizer.TYPE=Adam               <Adam> Adam optimizer.
--optimizer.beta1=0.9               <float> First moment decay. [default=0.9, from=Adam]
--optimizer.beta2=0.999             <float> Second moment decay. [default=0.999, from=Adam]
--optimizer.lr=???                  <float> Learning rate. [required, from=Optimizer]
--optimizer.weight_decay=0.0        <float> L2 regularization. [default=0.0, from=Optimizer]
```

The format uses:
- **First column**: `--dot.path=value` showing the CLI argument name and current/default value
- **Second column**: `<type> description [metadata]`
  - `<type>`: Parameter type in angle brackets
  - Description from docstring
  - `[metadata]`: Bracketed metadata showing:
    - `required` or `default=X`: Whether parameter is required or has a default value
    - `from=ClassName`: Which class originally defined this parameter (for inherited params)

**TYPE notation**: Object configuration groups use `.TYPE` suffix (e.g., `--optimizer.TYPE=Adam`) to clarify that TYPE is a special configuration key, distinguishing it from regular parameter values.

Special values:
- `???`: Missing required parameter (no value provided)
- `null`: Python `None` value
- `true`/`false`: Boolean values (YAML style)
- Strings shown without quotes (YAML style)

### Help with Current Configuration

You can pass current config data to show actual values instead of defaults:

```python
config_data = {
    "optimizer": {
        "TYPE": Adam,
        "lr": 0.001,
        "beta1": 0.95
    }
}
print(parser.help(config_data))
```

Output:
```
--optimizer.TYPE=Adam               <Adam> Adam optimizer.
--optimizer.beta1=0.95              <float> First moment decay. [default=0.9, from=Adam]
--optimizer.beta2=0.999             <float> Second moment decay. [default=0.999, from=Adam]
--optimizer.lr=0.001                <float> Learning rate. [required, from=Optimizer]
--optimizer.weight_decay=0.0        <float> L2 regularization. [default=0.0, from=Optimizer]
```

Notice how `lr` now shows `0.001` instead of `???`, and `beta1` shows the configured value `0.95` instead of the default `0.9`.

### Filtered Help

Use `--help.<key_path>` to show only arguments under a specific key path:

```bash
# Show only optimizer arguments
python your_script.py --help.optimizer

# Show only trainer arguments
python your_script.py --help.trainer

# Show only nested optimizer within trainer
python your_script.py --help.trainer.optimizer
```

Programmatically:

```python
# Filter to show only optimizer arguments
print(parser.help(config_data, key_path_filter="optimizer"))
```

This supports arbitrary nesting depth, allowing you to focus on specific configuration sections.

### Nested Object Configuration Expansion

When configuration includes nested objects with TYPE keys, the help system automatically expands them:

```python
config_data = {
    "trainer": {
        "TYPE": Trainer,
        "max_epochs": 100,
        "optimizer": {
            "TYPE": SGD,
            "lr": 0.01,
            "momentum": 0.9
        }
    }
}

print(parser.help(config_data))
```

Output:
```
--trainer.TYPE=Trainer                  <Trainer> Training controller.
--trainer.max_epochs=100                <int> Maximum training epochs. [required, from=Trainer]
--trainer.optimizer.TYPE=SGD            <SGD> Optimizer instance. [required, from=Trainer]
--trainer.optimizer.lr=0.01             <float> Learning rate. [required, from=Optimizer]
--trainer.optimizer.momentum=0.9        <float> Momentum factor. [default=0.0, from=SGD]
--trainer.optimizer.weight_decay=0.0    <float> Weight decay factor. [default=0.0, from=Optimizer]
```

Note that:
- Nested configs are fully expanded showing all their parameters
- The TYPE line for nested configs shows metadata from the parent (e.g., `[required, from=Trainer]`)
- Empty lines separate top-level configuration groups for readability
- Filtering works with nested paths: `--help.trainer.optimizer` shows only the nested optimizer

## Configuration Sources

SynConf supports multiple configuration sources that can be merged in any order:

### Source Types

1. **YAML files**: Files ending in `.yaml` or `.yml`
2. **CLI arguments**: Arguments starting with `--`

### Merging Order

Sources are merged left-to-right, with later sources overriding earlier ones:

```python
config = parser.parse(
    "base_config.yaml",    # Base settings
    "experiment.yaml",     # Experiment overrides
    "--optimizer.lr=0.01"  # CLI overrides
)
```

In this example:
1. `base_config.yaml` is loaded first
2. `experiment.yaml` overrides values from base
3. CLI argument `--optimizer.lr=0.01` overrides all previous `optimizer.lr` values

### Deep Merging

Nested dictionaries are merged recursively:

**base.yaml:**
```yaml
model:
  lr: 0.001
  layers: 12
```

**experiment.yaml:**
```yaml
model:
  lr: 0.01  # Overrides lr
  # layers: 12 is preserved
```

Result:
```yaml
model:
  lr: 0.01    # From experiment.yaml
  layers: 12  # From base.yaml
```

### CLI Argument Format

CLI arguments use dot notation for nested keys:

```bash
python train.py \
  --optimizer.TYPE=Adam \
  --optimizer.lr=0.001 \
  --optimizer.weight_decay=0.0001 \
  --model.layers=12 \
  --model.hidden_size=768
```

Values are parsed as YAML scalars, so you can use:
- Numbers: `--lr=0.001`
- Booleans: `--debug=true` or `--debug=false`
- Null: `--value=null`
- Strings: `--name=experiment`
- Lists: `--values=[1,2,3]`
- Dicts: `--config={key: value}`

## Next Steps

- Learn about [Validation](validation.md) for type checking and error handling
- Master [Interpolation](interpolation.md) for dynamic values and expressions
- Explore [Advanced Features](advanced.md) for REMOVE, LIST, and sweep functionality
