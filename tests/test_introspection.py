"""
Tests for: Introspection - Signature, Docstring, ParamSpec, CallableSpec, Kwargs Tracing
Corresponds to: src/synconf/introspection/
"""

from synconf.introspection.docstring import extract_docstring
from synconf.introspection.introspector import Introspector
from synconf.introspection.kwargs_tracer import KwargsTracer
from synconf.introspection.signature import extract_signature, get_callable_name
from synconf.introspection.spec import CallableSpec, ParamSpec

# === Test Fixtures ===


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


def create_optimizer(lr: float = 0.001, **kwargs) -> BaseOptimizer:
    """Factory to create an optimizer."""
    return Adam(lr=lr, **kwargs)


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


# === Signature Tests ===


def test_signature_extraction():
    """Tests for: Signature extraction - class params, has_var_keyword, function params."""
    # Test extract_signature for class with kwargs
    params, has_var_keyword = extract_signature(Adam)
    assert has_var_keyword is True
    param_names = [p.name for p in params]
    assert "beta1" in param_names and "beta2" in param_names

    # Test extract_signature for class without kwargs
    params, has_var_keyword = extract_signature(BaseOptimizer)
    assert has_var_keyword is False
    param_names = [p.name for p in params]
    assert "lr" in param_names and "weight_decay" in param_names

    # Test get_callable_name
    assert get_callable_name(Adam) == "Adam"
    assert get_callable_name(create_optimizer) == "create_optimizer"


# === Docstring Tests ===


def test_docstring_extraction():
    """Tests for: Docstring extraction - description, param descriptions, no docstring."""
    # Test extract_docstring
    desc, param_descs = extract_docstring(Adam)
    assert desc is not None
    assert "beta1" in param_descs and "beta2" in param_descs

    # Test no docstring
    def no_doc(x: int):
        pass

    desc, param_descs = extract_docstring(no_doc)
    assert desc is None and param_descs == {}


# === ParamSpec and CallableSpec Tests ===


def test_param_spec():
    """Tests for: ParamSpec - required, format_type, default handling."""
    param = ParamSpec(name="x", type_hint=int)
    assert param.is_required is True
    assert param.format_type() == "int"

    param_with_default = ParamSpec(name="y", type_hint=int, default=10)
    assert param_with_default.is_required is False

    param_no_hint = ParamSpec(name="z", type_hint=None)
    assert param_no_hint.format_type() == "Any"


def test_callable_spec():
    """Tests for: CallableSpec - is_class, is_function."""
    spec = CallableSpec(name="Adam", callable_obj=Adam)
    assert spec.is_class() is True and spec.is_function() is False

    spec_fn = CallableSpec(name="fn", callable_obj=create_optimizer)
    assert spec_fn.is_class() is False and spec_fn.is_function() is True


# === Introspector Tests ===


def test_introspector():
    """Tests for: Introspector - analyze, params, caching, clear_cache."""
    introspector = Introspector()

    # Test analyze
    spec = introspector.analyze(Adam)
    assert spec.name == "Adam" and spec.callable_obj == Adam
    assert "beta1" in spec.params

    # Test caching
    spec2 = introspector.analyze(Adam)
    assert spec is spec2  # Same object

    # Test clear_cache
    introspector.clear_cache()
    spec3 = introspector.analyze(Adam)
    assert spec is not spec3  # Different object


# === Kwargs Tracing Tests ===


def test_kwargs_tracing():
    """Tests for: Kwargs tracing - class hierarchy, function->class, function->function, class->function."""
    tracer = KwargsTracer()
    introspector = Introspector()

    # Test class -> parent class
    delegees = tracer.trace(Adam)
    assert len(delegees) == 1 and delegees[0] is BaseOptimizer

    # Test multi-level inheritance (AdamW -> Adam -> BaseOptimizer)
    delegees = tracer.trace(AdamW)
    assert len(delegees) == 1 and delegees[0] is Adam

    # Test no kwargs -> no delegees
    assert len(tracer.trace(BaseOptimizer)) == 0

    # Test function -> class
    delegees = tracer.trace(create_optimizer)
    assert len(delegees) >= 1  # AST finds Adam or BaseOptimizer

    # Test function -> function
    delegees = tracer.trace(outer_function)
    assert len(delegees) == 1 and delegees[0] is inner_function

    # Test class -> function
    delegees = tracer.trace(ProcessorClass)
    assert len(delegees) == 1 and delegees[0] is inner_function


def test_kwargs_param_collection():
    """Tests for: Kwargs param collection - delegee params, get_all_params, source tracking."""
    introspector = Introspector()

    # Test introspector collects params from delegees
    spec = introspector.analyze(Adam)
    assert "beta1" in spec.params and "beta2" in spec.params
    assert len(spec.kwargs_delegees) == 1

    # Test get_all_params includes delegee params
    all_params = spec.get_all_params()
    assert "beta1" in all_params and "lr" in all_params and "weight_decay" in all_params

    # Test multi-level get_all_params (AdamW -> Adam -> BaseOptimizer)
    spec = introspector.analyze(AdamW)
    all_params = spec.get_all_params()
    assert "amsgrad" in all_params  # AdamW
    assert "beta1" in all_params and "beta2" in all_params  # Adam
    assert "lr" in all_params and "weight_decay" in all_params  # BaseOptimizer

    # Test source tracking
    spec = introspector.analyze(Adam)
    all_params = spec.get_all_params()
    assert all_params["beta1"].source.name == "Adam"
    assert all_params["lr"].source.name == "BaseOptimizer"
