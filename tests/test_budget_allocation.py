"""Tests for model-controlled budget allocation in subcalls.

Tests cover:
1. _subcall with explicit budget reserves the specified amount (not all remaining)
2. _subcall_batched with budgets passes per-child budgets
3. _subcall_batched with budgets=None splits remaining budget equally
4. _rlm_query passes budget through to subcall_fn
5. _rlm_query_batched passes budgets through to subcall_batched_fn
6. _rlm_query_batched sequential fallback passes per-child budgets
7. Handler cost is synced into accumulator before code execution
"""

from unittest.mock import MagicMock, patch

import rlm.core.rlm as rlm_module
from rlm import RLM
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
# _subcall: budget parameter controls reservation amount
# ────────────────────────────────────────────────────────


class TestSubcallBudgetReservation:
    """Verify _subcall with explicit budget reserves only the requested amount."""

    def test_subcall_with_budget_reserves_specified_amount(self):
        """_subcall(prompt, budget=0.5) should reserve 0.5, not all remaining."""
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
                max_budget=1.0,
            )

            # Spy on reserve calls
            original_reserve = parent._cost_accumulator.reserve
            reserve_calls = []

            def tracking_reserve(amount):
                reserve_calls.append(amount)
                return original_reserve(amount)

            parent._cost_accumulator.reserve = tracking_reserve

            with patch.object(rlm_module, "RLM") as MockRLM:
                mock_child = MagicMock()
                mock_child.completion.return_value = _make_completion("done", cost=0.3)
                mock_child.close.return_value = None
                MockRLM.return_value = mock_child

                parent._subcall("test prompt", budget=0.5)

            # reserve() should have been called with 0.5, not 1.0
            assert len(reserve_calls) == 1
            assert reserve_calls[0] == 0.5

            parent.close()

    def test_subcall_without_budget_reserves_all_remaining(self):
        """_subcall(prompt) with budget=None should reserve all remaining."""
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
                max_budget=2.0,
            )

            original_reserve = parent._cost_accumulator.reserve
            reserve_calls = []

            def tracking_reserve(amount):
                reserve_calls.append(amount)
                return original_reserve(amount)

            parent._cost_accumulator.reserve = tracking_reserve

            with patch.object(rlm_module, "RLM") as MockRLM:
                mock_child = MagicMock()
                mock_child.completion.return_value = _make_completion("done", cost=0.1)
                mock_child.close.return_value = None
                MockRLM.return_value = mock_child

                parent._subcall("test prompt")  # budget=None

            # Should reserve all remaining (2.0)
            assert len(reserve_calls) == 1
            assert reserve_calls[0] == 2.0

            parent.close()


# ────────────────────────────────────────────────────────
# _subcall_batched: budget splitting and passthrough
# ────────────────────────────────────────────────────────


class TestSubcallBatchedBudgets:
    """Verify _subcall_batched handles budgets correctly."""

    def test_explicit_budgets_passed_to_children(self):
        """_subcall_batched with budgets=[0.3, 0.7] passes each to _subcall."""
        with patch.object(rlm_module, "get_client") as mock_gc:
            mock_lm = MagicMock()
            mock_lm.model_name = "test"
            mock_gc.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                max_depth=3,
                max_budget=1.0,
            )

            subcall_calls = []

            def mock_subcall(prompt, model=None, budget=None):
                subcall_calls.append({"prompt": prompt, "model": model, "budget": budget})
                return _make_completion(f"done: {prompt}", cost=0.1)

            parent._subcall = mock_subcall

            results = parent._subcall_batched(
                ["p1", "p2"], budgets=[0.3, 0.7]
            )

            assert len(results) == 2
            budgets_received = [c["budget"] for c in subcall_calls]
            assert sorted(budgets_received) == [0.3, 0.7]

            parent.close()

    def test_none_budgets_splits_equally(self):
        """_subcall_batched with budgets=None and max_budget splits equally."""
        with patch.object(rlm_module, "get_client") as mock_gc:
            mock_lm = MagicMock()
            mock_lm.model_name = "test"
            mock_gc.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                max_depth=3,
                max_budget=1.0,
            )

            subcall_calls = []

            def mock_subcall(prompt, model=None, budget=None):
                subcall_calls.append({"prompt": prompt, "budget": budget})
                return _make_completion(f"done: {prompt}", cost=0.1)

            parent._subcall = mock_subcall

            results = parent._subcall_batched(["a", "b", "c", "d"])

            assert len(results) == 4
            # Each should get 1.0 / 4 = 0.25
            for call in subcall_calls:
                assert call["budget"] == 0.25

            parent.close()

    def test_none_budgets_no_max_budget_passes_none(self):
        """Without max_budget, budgets=None means each child gets budget=None."""
        with patch.object(rlm_module, "get_client") as mock_gc:
            mock_lm = MagicMock()
            mock_lm.model_name = "test"
            mock_gc.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                max_depth=3,
                max_budget=None,  # No budget tracking
            )

            subcall_calls = []

            def mock_subcall(prompt, model=None, budget=None):
                subcall_calls.append({"budget": budget})
                return _make_completion(f"done: {prompt}")

            parent._subcall = mock_subcall

            parent._subcall_batched(["a", "b"])

            # No budget tracking → each child gets None
            for call in subcall_calls:
                assert call["budget"] is None

            parent.close()


# ────────────────────────────────────────────────────────
# LocalREPL: budget/budgets passthrough
# ────────────────────────────────────────────────────────


class TestREPLBudgetPassthrough:
    """Verify LocalREPL passes budget/budgets to subcall callbacks."""

    def test_rlm_query_passes_budget_to_subcall_fn(self):
        """rlm_query(prompt, budget=0.5) should pass budget to subcall_fn."""
        subcall_fn = MagicMock(return_value=_make_completion("ok"))

        repl = LocalREPL(subcall_fn=subcall_fn)
        result = repl.execute_code("answer = rlm_query('test', budget=0.5)")

        assert result.stderr == ""
        subcall_fn.assert_called_once_with("test", None, 0.5)
        repl.cleanup()

    def test_rlm_query_without_budget_passes_none(self):
        """rlm_query(prompt) should pass budget=None to subcall_fn."""
        subcall_fn = MagicMock(return_value=_make_completion("ok"))

        repl = LocalREPL(subcall_fn=subcall_fn)
        result = repl.execute_code("answer = rlm_query('test')")

        assert result.stderr == ""
        subcall_fn.assert_called_once_with("test", None, None)
        repl.cleanup()

    def test_rlm_query_batched_passes_budgets_to_batched_fn(self):
        """rlm_query_batched(prompts, budgets=[...]) passes budgets to subcall_batched_fn."""
        completions = [_make_completion(f"resp {i}") for i in range(2)]
        batched_fn = MagicMock(return_value=completions)

        repl = LocalREPL(subcall_batched_fn=batched_fn)
        result = repl.execute_code("answers = rlm_query_batched(['a', 'b'], budgets=[0.3, 0.7])")

        assert result.stderr == ""
        batched_fn.assert_called_once_with(["a", "b"], None, [0.3, 0.7])
        repl.cleanup()

    def test_rlm_query_batched_without_budgets_passes_none(self):
        """rlm_query_batched(prompts) passes budgets=None to subcall_batched_fn."""
        completions = [_make_completion(f"resp {i}") for i in range(2)]
        batched_fn = MagicMock(return_value=completions)

        repl = LocalREPL(subcall_batched_fn=batched_fn)
        result = repl.execute_code("answers = rlm_query_batched(['a', 'b'])")

        assert result.stderr == ""
        batched_fn.assert_called_once_with(["a", "b"], None, None)
        repl.cleanup()

    def test_rlm_query_batched_sequential_fallback_passes_budgets(self):
        """Sequential fallback passes budgets[i] to each subcall_fn call."""
        completions = [_make_completion(f"seq {i}") for i in range(2)]
        subcall_fn = MagicMock(side_effect=completions)

        repl = LocalREPL(subcall_fn=subcall_fn, subcall_batched_fn=None)
        result = repl.execute_code("answers = rlm_query_batched(['x', 'y'], budgets=[0.4, 0.6])")

        assert result.stderr == ""
        assert subcall_fn.call_count == 2
        calls = subcall_fn.call_args_list
        assert calls[0].args == ("x", None, 0.4) or calls[0] == (("x", None, 0.4),)
        assert calls[1].args == ("y", None, 0.6) or calls[1] == (("y", None, 0.6),)
        repl.cleanup()

    def test_rlm_query_batched_sequential_fallback_no_budgets(self):
        """Sequential fallback without budgets passes None to each subcall_fn."""
        completions = [_make_completion(f"seq {i}") for i in range(2)]
        subcall_fn = MagicMock(side_effect=completions)

        repl = LocalREPL(subcall_fn=subcall_fn, subcall_batched_fn=None)
        result = repl.execute_code("answers = rlm_query_batched(['x', 'y'])")

        assert result.stderr == ""
        calls = subcall_fn.call_args_list
        assert calls[0].args == ("x", None, None) or calls[0] == (("x", None, None),)
        assert calls[1].args == ("y", None, None) or calls[1] == (("y", None, None),)
        repl.cleanup()


# ────────────────────────────────────────────────────────
# Handler cost sync: accumulator reflects parent LLM spend
# ────────────────────────────────────────────────────────


class TestHandlerCostSync:
    """Verify handler cost is synced into accumulator before subcalls."""

    def test_sync_handler_cost_adds_delta_to_accumulator(self):
        """_sync_handler_cost should add only the delta since last sync."""
        with patch.object(rlm_module, "get_client") as mock_gc:
            mock_lm = MagicMock()
            mock_lm.model_name = "test"
            mock_gc.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                max_depth=3,
                max_budget=10.0,
            )

            # Mock handler reporting $2 cumulative cost
            mock_handler = MagicMock()
            mock_handler.get_usage_summary.return_value = UsageSummary(
                model_usage_summaries={
                    "test": ModelUsageSummary(
                        total_calls=1, total_input_tokens=100,
                        total_output_tokens=50, total_cost=2.0,
                    )
                }
            )

            parent._sync_handler_cost(mock_handler)

            # Accumulator should now have $2
            assert parent._cost_accumulator.total_cost == 2.0
            assert parent._last_handler_cost == 2.0

            # Second call with $3 cumulative — delta is $1
            mock_handler.get_usage_summary.return_value = UsageSummary(
                model_usage_summaries={
                    "test": ModelUsageSummary(
                        total_calls=2, total_input_tokens=200,
                        total_output_tokens=100, total_cost=3.0,
                    )
                }
            )

            parent._sync_handler_cost(mock_handler)

            # Accumulator should now have $3 total
            assert parent._cost_accumulator.total_cost == 3.0
            assert parent._last_handler_cost == 3.0

            parent.close()

    def test_remaining_budget_reflects_handler_cost(self):
        """After syncing handler cost, remaining_budget should be reduced."""
        with patch.object(rlm_module, "get_client") as mock_gc:
            mock_lm = MagicMock()
            mock_lm.model_name = "test"
            mock_gc.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                max_depth=3,
                max_budget=5.0,
            )

            assert parent._cost_accumulator.remaining_budget == 5.0

            mock_handler = MagicMock()
            mock_handler.get_usage_summary.return_value = UsageSummary(
                model_usage_summaries={
                    "test": ModelUsageSummary(
                        total_calls=1, total_input_tokens=100,
                        total_output_tokens=50, total_cost=2.0,
                    )
                }
            )

            parent._sync_handler_cost(mock_handler)

            # Remaining should be 5.0 - 2.0 = 3.0
            assert parent._cost_accumulator.remaining_budget == 3.0

            parent.close()

    def test_budget_check_includes_handler_and_child_costs(self):
        """_check_iteration_limits should see both handler and child costs.

        In normal flow, handler cost is synced into the accumulator in
        _completion_turn before _check_iteration_limits is called.
        """
        import pytest
        from rlm.core.types import CodeBlock, REPLResult, RLMIteration
        from rlm.utils.exceptions import BudgetExceededError

        with patch.object(rlm_module, "get_client") as mock_gc:
            mock_lm = MagicMock()
            mock_lm.model_name = "test"
            mock_gc.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test"},
                max_depth=3,
                max_budget=5.0,
            )

            # Simulate handler cost ($2) synced via _sync_handler_cost
            parent._cost_accumulator.add_cost(2.0)
            # Simulate child cost ($4) added in _subcall
            parent._cost_accumulator.add_cost(4.0)

            # Handler already synced, no new delta
            mock_handler = MagicMock()
            mock_handler.get_usage_summary.return_value = UsageSummary(
                model_usage_summaries={}
            )

            iteration = RLMIteration(
                prompt="test", response="code",
                code_blocks=[
                    CodeBlock(code="pass", result=REPLResult(stdout="", stderr="", locals={}))
                ],
            )

            # Total = $2 + $4 = $6 > $5 budget
            with pytest.raises(BudgetExceededError) as exc_info:
                parent._check_iteration_limits(iteration, 0, mock_handler)

            assert exc_info.value.spent == 6.0
            assert exc_info.value.budget == 5.0

            parent.close()
