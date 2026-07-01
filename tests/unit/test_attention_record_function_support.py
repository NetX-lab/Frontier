"""
Unit tests for AttentionWrapper RECORD_FUNCTION profile_method support.

These tests intentionally inspect source files instead of importing torch-heavy
profiling wrappers. The minimal simulator environment excludes torch/vllm/
flashinfer, while this contract must still be verifiable there.
"""

import ast
import inspect
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ATTENTION_WRAPPER_PATH = PROJECT_ROOT / "frontier/profiling/attention/attention_wrapper.py"
LINEAR_WRAPPER_PATH = PROJECT_ROOT / "frontier/profiling/linear_op/linear_op_wrapper.py"
MOE_WRAPPER_PATH = PROJECT_ROOT / "frontier/profiling/moe/moe_wrapper.py"
RECORD_FUNCTION_TRACER_PATH = PROJECT_ROOT / "frontier/profiling/utils/record_function_tracer.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _class_node(path: Path, class_name: str) -> ast.ClassDef:
    for node in _tree(path).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"Class {class_name} not found in {path}")


def _method_node(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    class_node = _class_node(path, class_name)
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"Method {class_name}.{method_name} not found in {path}")


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    return ast.get_source_segment(
        _source(path), _method_node(path, class_name, method_name)
    ) or ""


def _function_parameter_names(function_node: ast.FunctionDef) -> list[str]:
    return [arg.arg for arg in function_node.args.args]


def _parameter_default(function_node: ast.FunctionDef, parameter_name: str):
    args = function_node.args.args
    defaults = function_node.args.defaults
    default_offset = len(args) - len(defaults)
    for index, arg in enumerate(args):
        if arg.arg != parameter_name:
            continue
        default_index = index - default_offset
        if default_index < 0:
            return inspect.Parameter.empty
        return ast.literal_eval(defaults[default_index])
    raise AssertionError(f"Parameter {parameter_name} not found in {function_node.name}")


class TestAttentionWrapperProfileMethodParam:
    """Test that AttentionWrapper accepts profile_method parameter."""

    def test_constructor_accepts_profile_method_cuda_event(self):
        """Constructor should expose profile_method for backward-compatible usage."""
        init_node = _method_node(ATTENTION_WRAPPER_PATH, "AttentionWrapper", "__init__")

        assert "profile_method" in _function_parameter_names(init_node), (
            "AttentionWrapper.__init__ must accept profile_method parameter"
        )

    def test_constructor_accepts_output_dir(self):
        """output_dir parameter is required for RecordFunctionTracer trace export."""
        init_node = _method_node(ATTENTION_WRAPPER_PATH, "AttentionWrapper", "__init__")

        assert "output_dir" in _function_parameter_names(init_node), (
            "AttentionWrapper.__init__ must accept output_dir parameter"
        )

    def test_profile_method_default_is_record_function(self):
        """Default profile_method should be record_function (pure kernel time)."""
        init_node = _method_node(ATTENTION_WRAPPER_PATH, "AttentionWrapper", "__init__")
        default = _parameter_default(init_node, "profile_method")

        assert default == "record_function", (
            f"Default profile_method should be record_function, got {default}"
        )


class TestAttentionWrapperProfileMethodBranching:
    """Test that attention profiling methods have RECORD_FUNCTION branches."""

    def test_profile_method_has_record_function_branch(self):
        """profile() must contain RECORD_FUNCTION branching logic."""
        source = _method_source(ATTENTION_WRAPPER_PATH, "AttentionWrapper", "profile")
        assert "RECORD_FUNCTION" in source, (
            "AttentionWrapper.profile() must have RECORD_FUNCTION branch"
        )
        assert "RecordFunctionTracer" in source, (
            "AttentionWrapper.profile() must use RecordFunctionTracer"
        )

    def test_profile_mixed_has_record_function_branch(self):
        """profile_mixed() must contain RECORD_FUNCTION branching logic."""
        source = _method_source(ATTENTION_WRAPPER_PATH, "AttentionWrapper", "profile_mixed")
        assert "RECORD_FUNCTION" in source, (
            "AttentionWrapper.profile_mixed() must have RECORD_FUNCTION branch"
        )
        assert "RecordFunctionTracer" in source, (
            "AttentionWrapper.profile_mixed() must use RecordFunctionTracer"
        )

    def test_profile_true_mixed_has_record_function_branch(self):
        """profile_true_mixed() must contain RECORD_FUNCTION branching logic."""
        source = _method_source(
            ATTENTION_WRAPPER_PATH, "AttentionWrapper", "profile_true_mixed"
        )
        assert "RECORD_FUNCTION" in source, (
            "AttentionWrapper.profile_true_mixed() must have RECORD_FUNCTION branch"
        )
        assert "RecordFunctionTracer" in source, (
            "AttentionWrapper.profile_true_mixed() must use RecordFunctionTracer"
        )


class TestAttentionMainProfileMethodCLI:
    """Test that attention/main.py exposes --profile_method CLI argument."""

    def test_parse_args_has_profile_method(self):
        """parse_args() must include --profile_method argument."""
        from frontier.profiling.attention import main as attn_main

        source = inspect.getsource(attn_main.parse_args)
        assert "--profile_method" in source, (
            "attention/main.py parse_args() must include --profile_method argument"
        )

    def test_profile_method_choices_use_exportable_release_methods(self):
        """--profile_method choices must reject debug-only timing methods before GPU work."""
        from frontier.profiling.attention import main as attn_main

        source = inspect.getsource(attn_main.parse_args)
        assert "EXPORTABLE_PROFILE_METHOD_CHOICES" in source, (
            "attention/main.py must limit --profile_method to exportable release choices"
        )


class TestRecordFunctionTracerFailFast:
    """Test that RecordFunctionTracer raises on zero cuda_time."""

    def test_zero_cuda_time_raises_value_error(self):
        """Operations with zero CUDA kernel time must raise ValueError."""
        source = _method_source(
            RECORD_FUNCTION_TRACER_PATH,
            "RecordFunctionTracer",
            "get_operation_time_stats",
        )
        assert "raise ValueError" in source, (
            "RecordFunctionTracer must raise ValueError on zero cuda_time"
        )
        # Ensure the old skip-and-continue pattern is removed.
        assert "zero_time_ops" not in source, (
            "RecordFunctionTracer must not silently track zero-time ops"
        )


class TestAttentionWrapperDynamicZeroAllowlist:
    """Static contract tests for dynamic zero-CUDA allowlist in attention wrapper."""

    def test_dynamic_allowlist_adds_prefill_and_decode_when_phase_absent(self):
        """Inactive prefill/decode paths should be treated as valid zero-kernel ops."""
        source = _method_source(
            ATTENTION_WRAPPER_PATH,
            "AttentionWrapper",
            "_get_allow_zero_cuda_ops_for_current_forward",
        )

        assert "attn_input_reshape" in _source(ATTENTION_WRAPPER_PATH)
        assert "attn_output_reshape" in _source(ATTENTION_WRAPPER_PATH)
        assert 'getattr(attention_wrapper, "contains_prefill", True)' in source
        assert "_dense_profiling_op_name_by_role(" in source
        assert "AttentionOperatorRole.PREFILL_KERNEL" in source
        assert 'allowed_ops.add("attn_prefill")' not in source
        assert 'getattr(attention_wrapper, "contains_decode", True)' in source
        assert "AttentionOperatorRole.DECODE_KERNEL" in source
        assert 'allowed_ops.add("attn_decode")' not in source


class TestConsistencyAcrossModules:
    """Test that all three profiling modules have consistent RECORD_FUNCTION support."""

    def test_all_wrappers_accept_profile_method(self):
        """All wrapper classes must accept profile_method parameter."""
        wrappers = [
            ("AttentionWrapper", ATTENTION_WRAPPER_PATH),
            ("LinearOpWrapper", LINEAR_WRAPPER_PATH),
            ("MoEWrapper", MOE_WRAPPER_PATH),
        ]

        for class_name, path in wrappers:
            init_node = _method_node(path, class_name, "__init__")
            assert "profile_method" in _function_parameter_names(init_node), (
                f"{class_name}.__init__ must accept profile_method parameter"
            )

    def test_wrapper_profile_method_defaults_and_requirements(self):
        """Attention has a default while linear/moe keep profile_method required."""
        attention_init = _method_node(ATTENTION_WRAPPER_PATH, "AttentionWrapper", "__init__")
        assert _parameter_default(attention_init, "profile_method") == "record_function"

        linear_init = _method_node(LINEAR_WRAPPER_PATH, "LinearOpWrapper", "__init__")
        assert _parameter_default(linear_init, "profile_method") is inspect.Parameter.empty

        moe_init = _method_node(MOE_WRAPPER_PATH, "MoEWrapper", "__init__")
        assert _parameter_default(moe_init, "profile_method") is inspect.Parameter.empty
