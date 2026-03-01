"""Thread-safe cost accumulator shared across an entire RLM tree.

A single CostAccumulator instance is created by the root RLM and passed by
reference to every child/grandchild.  All cost mutations go through a lock
so concurrent children (from ThreadPoolExecutor in _subcall_batched) are safe.
"""

from __future__ import annotations

import threading


class CostAccumulator:
    """Tracks committed spend and budget reservations across all RLM descendants.

    Budget strategy for concurrent children:
    - Before spawning a child, the parent calls ``reserve(amount)`` which
      atomically moves ``amount`` from the remaining budget into
      ``_reserved_cost``.
    - When the child finishes, ``commit(actual, reservation)`` converts the
      reservation into committed cost and releases any unused portion.
    - ``remaining_budget`` accounts for both committed *and* reserved cost so
      concurrent siblings cannot double-spend the same budget.
    """

    def __init__(self, max_budget: float | None = None) -> None:
        self._lock = threading.Lock()
        self._total_cost: float = 0.0
        self._reserved_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_calls: int = 0
        self.max_budget: float | None = max_budget

    # -- read helpers (lock-free snapshots are fine for monitoring) ----------

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def remaining_budget(self) -> float | None:
        """Budget left after committed spend *and* outstanding reservations."""
        if self.max_budget is None:
            return None
        with self._lock:
            return self.max_budget - self._total_cost - self._reserved_cost

    def is_budget_exceeded(self) -> bool:
        if self.max_budget is None:
            return False
        return self._total_cost > self.max_budget

    # -- mutation methods ---------------------------------------------------

    def reserve(self, amount: float) -> float:
        """Reserve budget before spawning a child.

        Returns the *actual* amount reserved (capped at what is available).
        A return value of 0.0 means the budget is exhausted.
        """
        if self.max_budget is None:
            return amount  # no budget cap → grant whatever was asked

        with self._lock:
            available = self.max_budget - self._total_cost - self._reserved_cost
            granted = min(amount, max(available, 0.0))
            self._reserved_cost += granted
            return granted

    def commit(
        self,
        actual_cost: float,
        reservation: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        calls: int = 0,
    ) -> None:
        """Convert a reservation into committed cost after a child completes.

        Any unused portion of the reservation (``reservation - actual_cost``)
        is released back to the pool.  If ``actual_cost > reservation`` the
        overspend is still recorded — callers should check
        ``is_budget_exceeded()`` afterwards.
        """
        with self._lock:
            self._total_cost += actual_cost
            # Release the reservation (even if actual > reservation, we clamp
            # to avoid negative _reserved_cost)
            self._reserved_cost -= min(reservation, self._reserved_cost)
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_calls += calls

    def add_cost(
        self,
        cost: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        calls: int = 0,
    ) -> None:
        """Record the parent's own direct LLM cost (no reservation needed)."""
        with self._lock:
            self._total_cost += cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_calls += calls
