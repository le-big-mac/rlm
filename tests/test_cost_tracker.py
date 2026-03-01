"""Tests for CostAccumulator thread-safety and budget arithmetic."""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from rlm.core.cost_tracker import CostAccumulator


class TestCostAccumulatorBasics:
    """Basic single-threaded semantics."""

    def test_initial_state(self):
        acc = CostAccumulator(max_budget=10.0)
        assert acc.total_cost == 0.0
        assert acc.remaining_budget == 10.0
        assert not acc.is_budget_exceeded()

    def test_no_budget(self):
        acc = CostAccumulator(max_budget=None)
        assert acc.remaining_budget is None
        assert not acc.is_budget_exceeded()

    def test_add_cost(self):
        acc = CostAccumulator(max_budget=10.0)
        acc.add_cost(3.0, input_tokens=100, output_tokens=50, calls=1)
        assert acc.total_cost == 3.0
        assert acc.total_input_tokens == 100
        assert acc.total_output_tokens == 50
        assert acc.total_calls == 1
        assert acc.remaining_budget == 7.0

    def test_reserve_and_commit(self):
        acc = CostAccumulator(max_budget=10.0)
        granted = acc.reserve(4.0)
        assert granted == 4.0
        assert acc.remaining_budget == 6.0  # 10 - 0 committed - 4 reserved
        assert acc.total_cost == 0.0  # nothing committed yet

        # Commit less than reserved
        acc.commit(2.5, reservation=4.0, calls=1)
        assert acc.total_cost == 2.5
        assert acc.remaining_budget == 7.5  # 10 - 2.5 committed - 0 reserved

    def test_reserve_capped_at_remaining(self):
        acc = CostAccumulator(max_budget=5.0)
        acc.add_cost(3.0)
        granted = acc.reserve(10.0)  # ask for 10, only 2 available
        assert granted == 2.0
        assert acc.remaining_budget == 0.0

    def test_reserve_when_exhausted(self):
        acc = CostAccumulator(max_budget=5.0)
        acc.add_cost(5.0)
        granted = acc.reserve(1.0)
        assert granted == 0.0

    def test_overspend_detection(self):
        acc = CostAccumulator(max_budget=5.0)
        acc.reserve(5.0)
        acc.commit(7.0, reservation=5.0)  # spent more than reserved
        assert acc.is_budget_exceeded()
        assert acc.total_cost == 7.0

    def test_reserve_no_budget(self):
        """Without a budget cap, reserve returns the full amount asked."""
        acc = CostAccumulator(max_budget=None)
        granted = acc.reserve(999.0)
        assert granted == 999.0


class TestCostAccumulatorConcurrency:
    """Verify thread-safety under concurrent access."""

    def test_concurrent_commits(self):
        """N threads calling commit concurrently should not lose any cost."""
        acc = CostAccumulator(max_budget=None)
        n_threads = 50
        cost_per_thread = 1.0
        tokens_per_thread = 10

        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            acc.commit(
                cost_per_thread,
                reservation=0.0,
                input_tokens=tokens_per_thread,
                output_tokens=tokens_per_thread,
                calls=1,
            )

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert acc.total_cost == n_threads * cost_per_thread
        assert acc.total_input_tokens == n_threads * tokens_per_thread
        assert acc.total_output_tokens == n_threads * tokens_per_thread
        assert acc.total_calls == n_threads

    def test_concurrent_reserve_and_commit(self):
        """Concurrent reserves should not exceed the budget."""
        budget = 10.0
        acc = CostAccumulator(max_budget=budget)
        n_threads = 20

        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            granted = acc.reserve(1.0)
            with lock:
                results.append(granted)
            # Simulate work then commit
            acc.commit(granted, reservation=granted)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total granted should not exceed budget
        assert sum(results) <= budget + 1e-9
        # Total committed cost should equal sum of what was granted
        assert abs(acc.total_cost - sum(results)) < 1e-9

    def test_concurrent_add_cost(self):
        """Multiple threads calling add_cost concurrently."""
        acc = CostAccumulator(max_budget=1000.0)
        n_threads = 100

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(acc.add_cost, 1.0, 10, 5, 1) for _ in range(n_threads)]
            for f in as_completed(futures):
                f.result()

        assert acc.total_cost == float(n_threads)
        assert acc.total_input_tokens == n_threads * 10
        assert acc.total_output_tokens == n_threads * 5
        assert acc.total_calls == n_threads
