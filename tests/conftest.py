"""
Shared test fixtures for synconf tests.
"""


# === Base Classes ===


class BaseOptimizer:
    """Base optimizer with learning rate and weight decay."""

    def __init__(self, lr: float, weight_decay: float = 0.0):
        """Initialize optimizer.

        Args:
            lr: Learning rate.
            weight_decay: Weight decay factor.
        """
        self.lr = lr
        self.weight_decay = weight_decay


# Alias for backward compatibility in tests
Optimizer = BaseOptimizer


class Adam(BaseOptimizer):
    """Adam optimizer extends BaseOptimizer."""

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, **kwargs):
        """Initialize Adam.

        Args:
            beta1: First moment decay.
            beta2: Second moment decay.
        """
        super().__init__(**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2


class AdamW(Adam):
    """AdamW with amsgrad option."""

    def __init__(self, amsgrad: bool = False, **kwargs):
        """Initialize AdamW.

        Args:
            amsgrad: Whether to use AMSGrad variant.
        """
        super().__init__(**kwargs)
        self.amsgrad = amsgrad


# === Scheduler Classes ===


class Scheduler:
    """Learning rate scheduler."""

    def __init__(self, step_size: int = 10):
        """Initialize scheduler.

        Args:
            step_size: Step size for decay.
        """
        self.step_size = step_size


class StepLR(Scheduler):
    """Step learning rate scheduler."""

    def __init__(self, gamma: float = 0.1, **kwargs):
        """Initialize StepLR.

        Args:
            gamma: Decay factor.
        """
        super().__init__(**kwargs)
        self.gamma = gamma


class AdamWithScheduler(BaseOptimizer):
    """Adam variant that accepts a scheduler for deeply nested TYPE testing."""

    def __init__(self, scheduler: Scheduler = None, **kwargs):
        """Initialize AdamWithScheduler.

        Args:
            scheduler: Optional learning rate scheduler.
        """
        super().__init__(**kwargs)
        self.scheduler = scheduler


# === Trainer Classes ===


class Trainer:
    """Training controller."""

    def __init__(self, max_epochs: int, optimizer: BaseOptimizer | None = None):
        """Initialize trainer.

        Args:
            max_epochs: Maximum training epochs.
            optimizer: Optimizer instance.
        """
        self.max_epochs = max_epochs
        self.optimizer = optimizer


# === Callback Classes ===


class Callback:
    """Base callback."""

    def __init__(self, name: str):
        """Initialize callback.

        Args:
            name: Callback name.
        """
        self.name = name


class LoggingCallback(Callback):
    """Logging callback."""

    def __init__(self, level: str = "INFO", **kwargs):
        """Initialize logging callback.

        Args:
            level: Logging level.
        """
        super().__init__(**kwargs)
        self.level = level


class CallbackTrainer:
    """Trainer with callbacks for LIST testing."""

    def __init__(self, callbacks: list | None = None):
        """Initialize trainer.

        Args:
            callbacks: List of callbacks.
        """
        self.callbacks = callbacks or []

    @staticmethod
    def create_callback(name: str) -> "Callback":
        """Static method factory for creating callbacks.

        Args:
            name: Callback name.
        """
        return Callback(name=name)


# === Model Classes ===


class Model:
    """Test model for advanced features."""

    def __init__(self, lr: float, peft_config: dict | None = None):
        """Initialize model.

        Args:
            lr: Learning rate.
            peft_config: Optional PEFT configuration.
        """
        self.lr = lr
        self.peft_config = peft_config


class ConvModel:
    """Model class (not an optimizer)."""

    def __init__(self, layers: int):
        self.layers = layers


# === Experiment Classes ===


class Experiment:
    """Experiment configuration for sweep testing."""

    def __init__(self, seed: int = 0, name: str = "exp"):
        """Initialize experiment.

        Args:
            seed: Random seed.
            name: Experiment name.
        """
        self.seed = seed
        self.name = name


# === Function Fixtures ===


def create_optimizer(lr: float = 0.001, **kwargs) -> BaseOptimizer:
    """Factory to create an optimizer."""
    return Adam(lr=lr, **kwargs)


def create_logging_callback(name: str, level: str = "INFO") -> LoggingCallback:
    """Factory function for creating logging callbacks.

    Args:
        name: Callback name.
        level: Logging level.
    """
    return LoggingCallback(name=name, level=level)


def create_other() -> str:
    """Factory returning wrong type."""
    return "not an optimizer"


def inner_function(x: int, y: int = 0) -> int:
    """Inner function for delegation tests."""
    return x + y


def outer_function(a: str, **kwargs):
    """Outer function delegates to inner_function."""
    return inner_function(**kwargs)


class ProcessorClass:
    """Class that delegates to a function."""

    def __init__(self, name: str, **kwargs):
        inner_function(**kwargs)
        self.name = name
