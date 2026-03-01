# Variable and Expression Interpolation

SynConf supports dynamic value interpolation using the `~{...}` syntax. This allows you to reference other configuration values, evaluate Python expressions, and access environment variables.

## Motivation

Complex configurations often have:
- **Repeated values** that should stay in sync (e.g., learning rate used in multiple places)
- **Derived values** computed from other settings (e.g., effective batch size = batch_size × num_gpus)
- **Environment-dependent paths** (e.g., output directories relative to `$HOME`)

Interpolation eliminates duplication and makes configurations more maintainable and DRY (Don't Repeat Yourself).

## Syntax: `~{...}`

The `~{...}` syntax is:
- **YAML-compatible (unquoted)**: Works without quotes in YAML files
- **Bash-compatible**: Doesn't trigger shell variable expansion in CLI arguments
- **Clear and readable**: The tilde suggests "refers to" or "similar to"

SynConf automatically detects whether the content is:
- A simple variable reference (dotted path like `model.lr`)
- A Python expression (anything more complex like `batch_size * 4`)
- An environment variable (prefixed with `env:`)

## Basic Variable References

Reference other configuration values by their dotted path:

```yaml
base_lr: 0.001

optimizer:
  TYPE: Adam
  lr: ~{base_lr}              # References base_lr value
  weight_decay: 0.0

model:
  encoder:
    learning_rate: ~{base_lr}  # Same value, stays in sync
```

**Type preservation:** When a single interpolation covers the entire value, the original type is preserved:

```yaml
batch_size: 32                # int
effective_batch: ~{batch_size}  # Still int, not string
```

## Nested Path References

Use dot notation to reference nested values:

```yaml
optimizer:
  lr: 0.001
  momentum: 0.9

trainer:
  learning_rate: ~{optimizer.lr}        # Reference nested value
  initial_momentum: ~{optimizer.momentum}
```

## Expression Interpolation

Evaluate Python expressions to compute values:

```yaml
base_lr: 0.001
gpus: 4
batch_size: 32

optimizer:
  lr: ~{base_lr * 10}                    # Arithmetic: 0.01
  
training:
  effective_batch: ~{batch_size * gpus}  # Expression: 128
  warmup_steps: ~{1000 / gpus}           # Division: 250.0

model:
  layers: 12
  hidden_size: 768
  total_params: ~{model.layers * model.hidden_size}  # Nested access: 9216
```

**Supported in expressions:**
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Comparison: `<`, `>`, `<=`, `>=`, `==`, `!=`
- Logic: `and`, `or`, `not`
- Built-in functions: `len`, `int`, `float`, `str`, `bool`, `min`, `max`, `abs`, `round`, `sum`
- Nested dotted access: `model.layers`, `optimizer.lr`

**Security:** Expression evaluation is safe - imports, lambdas, and `__dunder__` access are not allowed.

## Partial/Embedded Interpolation

Embed interpolations within strings for concatenation:

```yaml
model_name: resnet50
experiment_id: exp_001

paths:
  # String concatenation with variables
  checkpoint: model_~{model_name}_~{experiment_id}.pt
  # Result: "model_resnet50_exp_001.pt"
  
  log_dir: logs/~{model_name}/run_~{experiment_id}
  # Result: "logs/resnet50/run_exp_001"
```

**YAML re-parsing:** After string substitution, the result is re-interpreted as a YAML scalar. This enables powerful type transformations:

```yaml
# Boolean construction
suffix: es
enabled: y~{suffix}        # "yes" → True (bool)
disabled: n~{suffix}       # "no" (invalid) → stays string

suffix2: o
is_active: n~{suffix2}     # "no" → False (bool)

# Number construction
digit: 0
port: 808~{digit}          # "8080" → 8080 (int)

decimal: 5
pi_approx: 3.1~{decimal}   # "3.15" → 3.15 (float)

# Null construction
rest: ull
value: n~{rest}            # "null" → None
```

This allows constructing any YAML type through string composition, making configurations extremely flexible.

## Environment Variables

Access environment variables with the `env:` prefix:

```yaml
paths:
  home: ~{env:HOME}                          # /Users/username
  data_dir: ~{env:HOME}/datasets             # Embedded in path
  cache: ~{env:XDG_CACHE_HOME}               # System-specific

experiment:
  output: ~{env:HOME}/experiments/~{model.name}  # Combined with config refs
```

If an environment variable is not set, an `UndefinedReferenceError` is raised with a clear error message.

## CLI Usage

Interpolations work seamlessly from the command line:

```bash
# Variable references (no quoting needed - bash-safe!)
python train.py --lr=~{base_lr}

# Expressions
python train.py --effective_batch=~{batch_size * gpus}

# Environment variables
python train.py --output_dir=~{env:HOME}/results

# Partial interpolation
python train.py --checkpoint=model_~{name}_best.pt
```

The `~{...}` syntax does not trigger bash variable expansion, so no quoting is necessary (unlike `${...}` which would be expanded by the shell).

## Chained References

References can chain through multiple levels:

```yaml
final_value: 100
intermediate: ~{final_value}
result: ~{intermediate}      # result = 100 (follows chain)

base_config:
  lr: 0.001

experiment:
  optimizer:
    learning_rate: ~{base_config.lr}  # Chains through nested paths
```

## Circular Reference Detection

SynConf detects circular references and raises a clear error:

```yaml
# ❌ This will raise CircularInterpolationError
a: ~{b}
b: ~{c}
c: ~{a}  # Error: Circular reference detected: a -> b -> c -> a
```

The error message shows the full cycle path to help you identify the issue.

## Cross-Source Interpolation

Interpolations work across different configuration sources:

**base.yaml:**
```yaml
base_lr: 0.001
```

**experiment.yaml:**
```yaml
optimizer:
  lr: ~{base_lr}  # References value from base.yaml
```

**CLI:**
```bash
python train.py base.yaml experiment.yaml --warmup_lr=~{base_lr}
```

After merging, all interpolations resolve correctly regardless of which source defined the referenced value.

## Resolution Order

Interpolations are resolved **after default imputation** to allow referencing defaults:

1. **Load** all sources (YAML files, CLI arguments)
2. **Merge** sources in order (later overrides earlier)
3. **Resolve TYPE keys** (strings → callables, including LIST conversion)
4. **Impute defaults** (fill in missing parameters with defaults)
5. **Resolve interpolations** (`~{...}` → actual values) ← Here
6. **Validate** against type hints

This order allows you to:
- Reference default values in interpolations
- Use interpolations in any parameter value
- Compute derived values from defaults

Example using defaults:

```yaml
optimizer:
  TYPE: Adam
  # Adam has default beta1=0.9
  adjusted_beta: ~{optimizer.beta1 + 0.05}  # Can reference default
```

## Best Practices

### 1. Define base values at the top level

```yaml
# ✓ Good - clear base values
base_lr: 0.001
batch_size: 32

optimizer:
  lr: ~{base_lr}
```

### 2. Use expressions for computed values

```yaml
# ✓ Good - makes relationship explicit
batch_size: 32
gpus: 4
effective_batch: ~{batch_size * gpus}

# ✗ Avoid - must manually calculate
effective_batch: 128  # What if batch_size changes?
```

### 3. Use environment variables for system paths

```yaml
# ✓ Good - portable across systems
data_dir: ~{env:HOME}/datasets

# ✗ Avoid - hardcoded path
data_dir: /Users/john/datasets
```

### 4. Name intermediate values clearly

```yaml
# ✓ Good - clear naming
model_name: bert
checkpoint_name: ~{model_name}_best.pt

# ✗ Avoid - hard to understand
x: bert
y: ~{x}_best.pt
```

## Error Handling

**Undefined reference:**
```yaml
value: ~{nonexistent_key}
# Error: Undefined reference: 'nonexistent_key'. Available keys: value
```

**Invalid expression:**
```yaml
value: ~{1 + * 2}
# Error: Invalid expression '~{1 + * 2}': syntax error
```

**Unsafe expression:**
```yaml
value: ~{__import__('os')}
# Error: Invalid expression: access to '__import__' not allowed
```

All errors include context about what went wrong and where, making debugging straightforward.

## Complete Example

Combining all interpolation features:

```yaml
# Base configuration
base_lr: 0.001
num_gpus: 4
model_name: transformer

# Paths using environment
paths:
  root: ~{env:HOME}/experiments
  output: ~{paths.root}/~{model_name}
  checkpoint: ~{paths.output}/checkpoint_~{num_gpus}gpu.pt

# Optimizer with expression
optimizer:
  TYPE: Adam
  lr: ~{base_lr}
  # Adaptive weight decay based on learning rate
  weight_decay: ~{base_lr / 10}

# Training config with computations
training:
  batch_size: 32
  # Effective batch size across GPUs
  effective_batch: ~{training.batch_size * num_gpus}
  # Warmup steps inversely proportional to GPUs
  warmup_steps: ~{int(1000 / num_gpus)}
  # Total steps computed from data size
  total_steps: ~{100000 // training.effective_batch}

# Model with derived values
model:
  TYPE: Transformer
  layers: 12
  hidden_size: 768
  # Compute total parameters
  total_params: ~{model.layers * model.hidden_size}
  
# Flags using YAML re-parsing
debug:
  suffix: es
  verbose: y~{debug.suffix}    # True
  profiling: n~{debug.suffix}  # "no" + "es" = "noes" (stays string)
```

## Next Steps

- Learn about [Advanced Features](advanced.md) for REMOVE, LIST, and sweep functionality
- Review [Validation](validation.md) for type checking and error handling
- Explore [Core Concepts](core-concepts.md) for TYPE system and help generation
