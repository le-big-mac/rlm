"""Tests for concurrent rlm_query_batched and the line-494 cost bug fix."""

import time
from unittest.mock import MagicMock, patch

import rlm.core.rlm as rlm_module
from rlm import RLM
from rlm.core.cost_tracker import CostAccumulator
from rlm.core.types import (
    ModelUsageSummary,
    RLMChatCompletion,
    UsageSummary,
)
from rlm.environments.local_repl import LocalREPL


def _make_completion(response: str, cost: float = 0.0) -> RLMChatCompletion:
    return RLMChatCompletion(
        root_model="test-model",
        prompt="test",
        response=response,
        usage_summary=UsageSummary(
            model_usage_summaries={
                "test-model": ModelUsageSummary(
                    total_calls=1,
                    total_input_tokens=10,
                    total_output_tokens=5,
                    total_cost=cost,
                )
            }
        ),
        execution_time=0.1,
    )


# ────────────────────────────────────────────────────────
# Tests for subcall_batched_fn in LocalREPL
# ────────────────────────────────────────────────────────


class TestBatchedSubcallFnInREPL:
    """Verify LocalREPL prefers subcall_batched_fn over sequential subcall_fn."""

    def test_uses_batched_fn_when_available(self):
        completions = [_make_completion(f"resp {i}") for i in range(3)]
        batched_fn = MagicMock(return_value=completions)
        subcall_fn = MagicMock()  # should NOT be called

        repl = LocalREPL(subcall_fn=subcall_fn, subcall_batched_fn=batched_fn)
        result = repl.execute_code("answers = rlm_query_batched(['a', 'b', 'c'])")

        assert result.stderr == ""
        assert repl.locals["answers"] == ["resp 0", "resp 1", "resp 2"]
        batched_fn.assert_called_once()
        subcall_fn.assert_not_called()
        assert len(result.rlm_calls) == 3
        repl.cleanup()

    def test_falls_back_to_sequential_when_no_batched_fn(self):
        completions = [_make_completion(f"seq {i}") for i in range(2)]
        subcall_fn = MagicMock(side_effect=completions)

        repl = LocalREPL(subcall_fn=subcall_fn, subcall_batched_fn=None)
        result = repl.execute_code("answers = rlm_query_batched(['x', 'y'])")

        assert result.stderr == ""
        assert repl.locals["answers"] == ["seq 0", "seq 1"]
        assert subcall_fn.call_count == 2
        repl.cleanup()

    def test_batched_fn_with_model_override(self):
        completions = [_make_completion("ok")]
        batched_fn = MagicMock(return_value=completions)

        repl = LocalREPL(subcall_batched_fn=batched_fn)
        repl.execute_code("rlm_query_batched(['q1'], model='custom')")

        batched_fn.assert_called_once_with(["q1"], "custom")
        repl.cleanup()

    def test_batched_fn_error_returns_error_strings(self):
        batched_fn = MagicMock(side_effect=RuntimeError("batch boom"))

        repl = LocalREPL(subcall_batched_fn=batched_fn)
        result = repl.execute_code("answers = rlm_query_batched(['a', 'b'])")

        assert result.stderr == ""
        answers = repl.locals["answers"]
        assert len(answers) == 2
        assert all("Error" in a for a in answers)
        assert all("batch boom" in a for a in answers)
        repl.cleanup()


# ────────────────────────────────────────────────────────
# Integration: concurrent execution actually overlaps
# ────────────────────────────────────────────────────────


class TestConcurrentExecution:
    """Verify _subcall_batched runs children in parallel."""

    def test_parallel_wall_time(self):
        """With mocked _subcall that sleeps, wall time should be < sum of sleeps."""
        sleep_time = 0.3
        n_prompts = 4

        with patch.object(rlm_module, "get_client") as mock_gc:
            mock_lm = MagicMock()
            mock_lm.model_name = "test"
            mock_lm.get_usage_summary.return_value = UsageSummary(model_usage_summaries={})
            mock_gc.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                max_depth=3,
            )

            def slow_subcall(prompt, model=None):
                time.sleep(sleep_time)
                return _make_completion(f"done: {prompt}")

            parent._subcall = slow_subcall

            start = time.perf_counter()
            results = parent._subcall_batched(
                [f"prompt_{i}" for i in range(n_prompts)]
            )
            elapsed = time.perf_counter() - start

            assert len(results) == n_prompts
            assert all("done:" in r.response for r in results)
            # If parallel, elapsed should be roughly sleep_time, not n*sleep_time
            assert elapsed < sleep_time * n_prompts * 0.7

            parent.close()


# ────────────────────────────────────────────────────────
# Regression: line 494 bug fix
# ────────────────────────────────────────────────────────


class TestCumulativeCostIncludesChildren:
    """_cumulative_cost must include both handler cost and child (accumulator) cost."""

    def test_cumulative_cost_after_child_subcall(self):
        """After a child reports cost via the accumulator, _check_iteration_limits
        should compute _cumulative_cost = handler_cost + accumulator.total_cost."""
        with patch.object(rlm_module, "get_client") as mock_gc:
            mock_lm = MagicMock()
            mock_lm.model_name = "test"
            # Handler reports $2 cost
            mock_lm.get_usage_summary.return_value = UsageSummary(
                model_usage_summaries={
                    "test": ModelUsageSummary(
                        total_calls=1,
                        total_input_tokens=100,
                        total_output_tokens=50,
                        total_cost=2.0,
                    )
                }
            )
            mock_gc.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                max_depth=3,
                max_budget=100.0,
            )

            # Simulate child cost via the accumulator
            parent._cost_accumulator.add_cost(5.0)

            # Create a dummy iteration with no errors
            from rlm.core.types import CodeBlock, REPLResult, RLMIteration

            dummy_iter = RLMIteration(
                prompt="x",
                response="y",
                code_blocks=[
                    CodeBlock(
                        code="pass",
                        result=REPLResult(stdout="", stderr="", locals={}),
                    )
                ],
            )

            from rlm.core.lm_handler import LMHandler

            handler = MagicMock(spec=LMHandler)
            handler.get_usage_summary.return_value = mock_lm.get_usage_summary.return_value

            parent._check_iteration_limits(dummy_iter, 0, handler)

            # _cumulative_cost should be handler(2) + accumulator(5) = 7
            assert parent._cumulative_cost == 7.0

            parent.close()

    def test_budget_exceeded_with_child_cost(self):
        """Budget should be exceeded when handler + child cost > max_budget."""
        with patch.object(rlm_module, "get_client") as mock_gc:
            mock_lm = MagicMock()
            mock_lm.model_name = "test"
            # Handler reports $3
            mock_lm.get_usage_summary.return_value = UsageSummary(
                model_usage_summaries={
                    "test": ModelUsageSummary(
                        total_calls=1,
                        total_input_tokens=100,
                        total_output_tokens=50,
                        total_cost=3.0,
                    )
                }
            )
            mock_gc.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                max_depth=3,
                max_budget=5.0,
            )

            # Child spent $4 via accumulator → total = 3 + 4 = 7 > 5
            parent._cost_accumulator.add_cost(4.0)

            from rlm.core.types import CodeBlock, REPLResult, RLMIteration
            from rlm.utils.exceptions import BudgetExceededError

            dummy_iter = RLMIteration(
                prompt="x",
                response="y",
                code_blocks=[
                    CodeBlock(
                        code="pass",
                        result=REPLResult(stdout="", stderr="", locals={}),
                    )
                ],
            )

            from rlm.core.lm_handler import LMHandler

            handler = MagicMock(spec=LMHandler)
            handler.get_usage_summary.return_value = mock_lm.get_usage_summary.return_value

            import pytest

            with pytest.raises(BudgetExceededError):
                parent._check_iteration_limits(dummy_iter, 0, handler)

            parent.close()


# ────────────────────────────────────────────────────────
# CostAccumulator is shared across tree
# ────────────────────────────────────────────────────────


class TestAccumulatorSharedInTree:
    """Verify child RLM receives the same CostAccumulator."""

    def test_child_gets_parent_accumulator(self):
        captured = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured["accumulator"] = kwargs.get("_cost_accumulator")
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_gc:
            mock_lm = MagicMock()
            mock_lm.model_name = "test"
            mock_lm.completion.return_value = "FINAL(answer)"
            mock_lm.get_usage_summary.return_value = UsageSummary(model_usage_summaries={})
            mock_lm.get_last_usage.return_value = UsageSummary(model_usage_summaries={})
            mock_gc.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                max_depth=3,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("hello")

            assert captured["accumulator"] is parent._cost_accumulator

            parent.close()
