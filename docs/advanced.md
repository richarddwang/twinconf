# Advanced Features

Learn about SynConf's advanced features: REMOVE during merging, LIST specification, and parameter sweeps.

## REMOVE During Merging

When merging configurations, you can use the special string `"REMOVE"` to delete keys from previous configs.

### Basic Usage

**base.yaml:**
```yaml
model:
  lr: 0.1
  peft_config:
    TYPE: LoRaConfig
    r: 2
    alpha: 4
```

**Override with CLI:**
```bash
python train.py base.yaml --model.peft_config=REMOVE
```

**Result:**
```yaml
model:
  lr: 0.1
  # peft_config is removed
```

### Use Cases

**1. Removing Optional Components:**
```yaml
# base.yaml - includes fancy features
model:
  layers: 12
  attention:
    TYPE: MultiHeadAttention
    heads: 8
  cache_config:
    TYPE: KVCache
    size: 1024

# simple.yaml - disable caching
model:
  cache_config: REMOVE
```

**2. Simplifying Nested Configs:**
```yaml
# full.yaml - complete config
trainer:
  optimizer:
    TYPE: Adam
    lr: 0.001
  scheduler:
    TYPE: CosineSchedule
    warmup: 1000

# no_scheduler.yaml - remove scheduler
trainer:
  scheduler: REMOVE
```

**3. Cleaning Up Experiments:**
```bash
# Remove debug settings for production
python train.py base.yaml debug.yaml --debug=REMOVE --verbose=REMOVE
```

### Behavior

- `REMOVE` only affects keys that exist in previous configs
- If the key doesn't exist, `REMOVE` has no effect
- REMOVE is processed during the merge step, before any resolution
- Works at any nesting level

### Example

Use `REMOVE` in YAML or pass `--key=REMOVE` from the CLI:

```yaml
# base.yaml
model:
  lr: 0.1
  feature_x: true

# override.yaml — removes feature_x during merge
model:
  feature_x: REMOVE
```

```bash
# CLI removal
python train.py base.yaml --model.feature_x=REMOVE
```

## LIST Specification

Use `TYPE: LIST` to convert a dictionary into a list, enabling dict-based list specification and modification.

### Why LIST?

Lists in YAML are great for ordering, but dicts are better for:
- **Named elements** - easier to identify what each item is
- **Selective overriding** - modify one element without redefining all
- **Merging** - override specific elements across configs

LIST gives you the flexibility of dicts with the semantics of lists.

### Basic Usage

**config.yaml:**
```yaml
trainer:
  callbacks:
    TYPE: LIST
    log:
      TYPE: LoggingCallback
      level: INFO
    prec:
      TYPE: PrecisionCallback
      precision: fp16
```

After resolution, `trainer.callbacks` becomes a list:

```python
[
  LoggingCallback(level="INFO"),
  PrecisionCallback(precision="fp16")
]
```

### Merging Lists

Override specific elements by key:

**base.yaml:**
```yaml
trainer:
  callbacks:
    TYPE: LIST
    log:
      TYPE: LoggingCallback
      level: INFO
    prec:
      TYPE: PrecisionCallback
      precision: fp16
    checkpoint:
      TYPE: CheckpointCallback
      every: 1000
```

**experiment.yaml:**
```yaml
trainer:
  callbacks:
    log:
      TYPE: CustomLoggingCallback  # Override just the log callback
      level: DEBUG
```

**Result (after merge and resolution):**
```python
[
  CustomLoggingCallback(level="DEBUG"),  # Modified
  PrecisionCallback(precision="fp16"),    # Unchanged
  CheckpointCallback(every=1000)          # Unchanged
]
```

### Element Ordering

Elements are ordered by dictionary insertion order (Python 3.7+ guarantees this):

```yaml
callbacks:
  TYPE: LIST
  third:
    TYPE: ThirdCallback
  first:
    TYPE: FirstCallback
  second:
    TYPE: SecondCallback
```

Results in:
```python
[ThirdCallback(), FirstCallback(), SecondCallback()]
```

To control order, use explicit ordering in your base config.

### Use Cases

**1. Plugin Systems:**
```yaml
plugins:
  TYPE: LIST
  auth:
    TYPE: AuthPlugin
  logging:
    TYPE: LoggingPlugin
  monitoring:
    TYPE: MonitoringPlugin
```

**2. Processing Pipelines:**
```yaml
transforms:
  TYPE: LIST
  normalize:
    TYPE: Normalize
    mean: [0.485, 0.456, 0.406]
  resize:
    TYPE: Resize
    size: 224
  augment:
    TYPE: RandomAugment
```

**3. Multi-Stage Training:**
```yaml
stages:
  TYPE: LIST
  pretrain:
    epochs: 10
    lr: 0.001
  finetune:
    epochs: 5
    lr: 0.0001
```

### Removing List Elements

Combine LIST with REMOVE to remove elements:

```yaml
# base.yaml
callbacks:
  TYPE: LIST
  log: {...}
  checkpoint: {...}
  early_stop: {...}

# experiment.yaml - remove early stopping
callbacks:
  early_stop: REMOVE
```

## Parameter Sweeps

Run multiple configurations with Cartesian product sweeps using the `sweep` subcommand.

### Basic Usage

```bash
python train.py config.yaml sweep --model.lr=[1e-4,2e-5] --exp.seed=[0,1,2]
```

This generates 2 × 3 = 6 configurations:
1. `lr=1e-4, seed=0`
2. `lr=1e-4, seed=1`
3. `lr=1e-4, seed=2`
4. `lr=2e-5, seed=0`
5. `lr=2e-5, seed=1`
6. `lr=2e-5, seed=2`

### How It Works

1. **Load and merge** base configuration from sources before `sweep`
2. **Generate** all combinations (Cartesian product)
3. **Apply** each combination to a copy of base config
4. **Resolve and validate** each config independently
5. **Return** list of Config objects

### Argument Order Matters

The order of sweep arguments defines the loop order:

```bash
# Outer loop: lr, Inner loop: seed
sweep --model.lr=[1e-4,2e-5] --exp.seed=[0,1,2]

# Generates:
# lr=1e-4, seed=0
# lr=1e-4, seed=1
# lr=1e-4, seed=2
# lr=2e-5, seed=0
# lr=2e-5, seed=1
# lr=2e-5, seed=2
```

### Full Example

**config.yaml:**
```yaml
model:
  lr: 3e-4
  wd: 1e-2
  
exp:
  seed: 3
  name: baseline
```

**Command:**
```bash
python train.py config.yaml sweep --model.lr=[1e-4,2e-5] --exp.seed=[0,1,2]
```

**Generated Configs:**

Config 1:
```yaml
model:
  lr: 1e-4   # Swept
  wd: 1e-2   # From base
exp:
  seed: 0    # Swept
  name: baseline  # From base
```

Config 5:
```yaml
model:
  lr: 2e-5   # Swept
  wd: 1e-2   # From base
exp:
  seed: 1    # Swept
  name: baseline  # From base
```

### Using Sweep Results

```python
from synconf import SynConfParser

parser = SynConfParser()
parser.add("model", Model)
parser.add("exp", Experiment)

# parse_args returns list of configs when sweep is used
configs = parser.parse_args()  # From sys.argv

if isinstance(configs, list):
    # Sweep mode - run multiple configs
    for i, config in enumerate(configs):
        print(f"Running config {i+1}/{len(configs)}")
        model = config.model.init()
        exp = config.exp.init()
        train(model, exp)
else:
    # Single config mode
    model = configs.model.init()
    exp = configs.exp.init()
    train(model, exp)
```

### Validation Benefits

All configurations are validated **before** any experiments run:

```bash
python train.py config.yaml sweep --model.lr=[1e-4,2e-5,invalid] --seed=[0,1,2]
# ValueError: Configuration validation failed:
#   Invalid type for 'model.lr': expected float, got str 'invalid' [source: CLI, owner: Model]

# No experiments ran - failed fast
```

This prevents:
- Wasting compute on invalid configs
- Discovering errors mid-sweep
- Partial experiment runs

### Value Format

Sweep values are parsed as YAML lists:

```bash
# Numbers
--lr=[0.001,0.01,0.1]

# Strings
--optimizer=[Adam,SGD,RMSprop]

# Booleans
--debug=[true,false]

# Mixed types (if parameter accepts multiple types)
--value=[1,2.5,null,true]

# Nested values
--config=[{a: 1},{a: 2}]
```

### Use Cases

**1. Hyperparameter Tuning:**
```bash
python train.py base.yaml sweep \
  --model.lr=[1e-4,5e-5,1e-5] \
  --model.dropout=[0.1,0.2,0.3] \
  --model.layers=[6,12,24]
# 3 × 3 × 3 = 27 experiments
```

**2. Reproducibility:**
```bash
python train.py config.yaml sweep --exp.seed=[0,1,2,3,4]
# Run 5 random seeds for statistical significance
```

**3. Architecture Search:**
```bash
python train.py base.yaml sweep \
  --model.TYPE=[ConvNet,ResNet,Transformer] \
  --model.width=[128,256,512]
# 3 × 3 = 9 architectures
```

**4. Ablation Studies:**
```bash
python train.py full.yaml sweep \
  --use_feature_x=[true,false] \
  --use_feature_y=[true,false] \
  --use_feature_z=[true,false]
# 2³ = 8 ablations
```

### Best Practices

**1. Start Small:**
```bash
# Test with small sweep first
python train.py config.yaml sweep --lr=[1e-4,1e-5] --seed=[0]

# Then scale up
python train.py config.yaml sweep --lr=[1e-4,5e-5,1e-5] --seed=[0,1,2,3,4]
```

**2. Validate Configs First:**
```python
# Check all configs before running
configs = parser.parse_args()  
print(f"Generated {len(configs)} configurations")

for i, config in enumerate(configs):
    print(f"Config {i+1}: lr={config.model.lr}, seed={config.exp.seed}")

# Optionally exit here to review configs before running
```

**3. Log Sweep Parameters:**
```python
for i, config in enumerate(configs):
    exp_name = f"sweep_{i}_lr{config.model.lr}_seed{config.exp.seed}"
    # Use unique names for each config
```

**4. Consider Compute Costs:**
```bash
# 5 × 5 × 5 = 125 configs might be too many
sweep --lr=[...5 values...] --wd=[...5 values...] --bs=[...5 values...]

# Consider smaller sweeps or staged sweeps
```

## Combining Advanced Features

You can combine REMOVE, LIST, and sweep:

**base.yaml:**
```yaml
model:
  callbacks:
    TYPE: LIST
    log:
      TYPE: LoggingCallback
    debug:
      TYPE: DebugCallback
```

**Command:**
```bash
python train.py base.yaml sweep \
  --model.callbacks.debug=REMOVE \
  --model.lr=[0.001,0.01]
```

This removes the debug callback and sweeps over learning rates.

## Next Steps

- Master [Interpolation](interpolation.md) for dynamic values and expressions
- Review [Validation](validation.md) for type checking and error handling
- Learn [Core Concepts](core-concepts.md) for initialization and TYPE system
- Check [Quick Start](quickstart.md) for basic usage examples
