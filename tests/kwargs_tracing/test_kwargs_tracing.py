"""
Tests for: Kwargs Tracing
Corresponds to: docs/core-concepts.md#help-generation (from=Source metadata)
"""

from conftest import Adam

from synconf import SynConfParser

# === Test fixtures: must be at module level for AST kwargs tracing to resolve globals ===


def _build_model(hidden_size: int, dropout: float = 0.1):
    """Build a model config dict."""
    return {"hidden_size": hidden_size, "dropout": dropout}


def _create_experiment(name: str = "exp", **kwargs):
    """Create experiment, delegating kwargs to _build_model."""
    return {"name": name, "model": _build_model(**kwargs)}


class _Pipeline:
    """Pipeline that delegates kwargs to its _build_model method."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = self._build_model(**kwargs)

    def _build_model(self, layers: int = 3, hidden: int = 256):
        """Build a model config."""
        return {"layers": layers, "hidden": hidden}


class _Tokenizer:
    """Tokenizer with vocab_size parameter."""

    def __init__(self, vocab_size: int = 30000, lowercase: bool = True):
        self.vocab_size = vocab_size
        self.lowercase = lowercase

    @classmethod
    def from_config(cls, vocab_size: int = 30000, lowercase: bool = True):
        """Create tokenizer from config."""
        return cls(vocab_size=vocab_size, lowercase=lowercase)


class _TextProcessor:
    """Processor that delegates kwargs to _Tokenizer.from_config classmethod."""

    def __init__(self, max_length: int = 512, **kwargs):
        self.max_length = max_length
        self.tokenizer = _Tokenizer.from_config(**kwargs)


def _configure_logging(log_level: str = "INFO", log_file: str | None = None):
    """Configure logging settings."""
    return {"level": log_level, "file": log_file}


class _Application:
    """App that delegates kwargs to _configure_logging function."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.logging = _configure_logging(**kwargs)


# === Tests ===


def test_class_hierarchy_kwargs_tracing():
    """Tests for: class __init__ delegation via super().__init__(**kwargs) across multiple levels."""

    class AdamW(Adam):
        """AdamW with amsgrad option."""

        def __init__(self, amsgrad: bool = False, **kwargs):
            super().__init__(**kwargs)
            self.amsgrad = amsgrad

    parser = SynConfParser()
    parser.add("optimizer", AdamW)

    # All params from the chain (AdamW -> Adam -> BaseOptimizer) should be configurable
    config = parser.parse(
        "--optimizer.lr=0.01",
        "--optimizer.beta1=0.95",
        "--optimizer.amsgrad=true",
    )
    optimizer = config.optimizer.init()
    assert isinstance(optimizer, AdamW)
    assert optimizer.lr == 0.01  # from BaseOptimizer
    assert optimizer.beta1 == 0.95  # from Adam
    assert optimizer.amsgrad is True  # from AdamW


def test_function_kwargs_tracing():
    """Tests for: function-to-function delegation via name(**kwargs)."""
    parser = SynConfParser()
    parser.add("experiment", _create_experiment)

    # 'hidden_size' from _build_model should be accessible through experiment
    config = parser.parse("--experiment.name=test", "--experiment.hidden_size=512")
    result = config.experiment.init()
    assert result["name"] == "test"
    assert result["model"]["hidden_size"] == 512

    # Help shows parameter source
    help_text = parser.help()
    assert "from=_build_model" in help_text


def test_instance_method_kwargs_tracing():
    """Tests for: instance method delegation via self.method(**kwargs)."""
    parser = SynConfParser()
    parser.add("pipeline", _Pipeline)

    config = parser.parse("--pipeline.name=main", "--pipeline.layers=5", "--pipeline.hidden=128")
    pipeline = config.pipeline.init()
    assert pipeline.name == "main"
    assert pipeline.model["layers"] == 5
    assert pipeline.model["hidden"] == 128


def test_classmethod_kwargs_tracing():
    """Tests for: class __init__ delegating kwargs to a @classmethod factory."""
    parser = SynConfParser()
    parser.add("processor", _TextProcessor)

    # vocab_size from _Tokenizer.from_config should be accessible through processor
    config = parser.parse("--processor.max_length=256", "--processor.vocab_size=50000")
    processor = config.processor.init()
    assert processor.max_length == 256
    assert processor.tokenizer.vocab_size == 50000


def test_mixed_class_function_kwargs_tracing():
    """Tests for: class __init__ delegating kwargs to a standalone function."""
    parser = SynConfParser()
    parser.add("app", _Application)

    config = parser.parse("--app.name=myapp", "--app.log_level=DEBUG")
    app = config.app.init()
    assert app.name == "myapp"
    assert app.logging["level"] == "DEBUG"

    # Help shows function as source of delegated params
    help_text = parser.help()
    assert "from=_configure_logging" in help_text
