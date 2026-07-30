"""Shadow API execution primitives for SimGuard.

Adapters provide a concrete executor for their runtime/environment. SimGuard
owns the sequence: shadow execute -> observe result/diff -> judge -> commit or
block. This keeps the defense portable across AgentDojo, SmolAgents, Voyager,
ExpeL, and real API proxies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ShadowExecutionResult:
    """Observed result of running a proposed tool call in a shadow environment."""

    result: str
    error: str | None = None
    state_diff: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardedToolCallResult:
    """Result returned after SimGuard decides whether to commit a tool call."""

    trace: Any
    result: str
    error: str | None
    committed: bool
    shadow: ShadowExecutionResult


class ShadowToolExecutor(Protocol):
    """Runtime-specific bridge used by SimGuard to execute shadow APIs."""

    def shadow_execute(self, tool_name: str, tool_arguments: dict) -> ShadowExecutionResult:
        """Run the tool against a shadow/sandbox/snapshot environment."""

    def commit(self, tool_name: str, tool_arguments: dict) -> tuple[str, str | None]:
        """Replay the approved tool call against the live environment."""


class RuntimeShadowToolExecutor:
    """Generic executor for Pydantic-like env + runtime.run_function systems.

    This matches AgentDojo directly, but is intentionally not AgentDojo-specific:
    any agent runtime with ``run_function(env, tool_name, args)`` and an env that
    supports ``model_copy(deep=True)``/``model_dump(mode="json")`` can use it.
    """

    def __init__(
        self,
        runtime: Any,
        env: Any,
        result_formatter: Callable[[Any], str] | None = None,
    ) -> None:
        self.runtime = runtime
        self.env = env
        self.result_formatter = result_formatter or str

    def _format(self, result: Any) -> str:
        try:
            return self.result_formatter(result)
        except Exception:
            return str(result)

    def _copy_env(self) -> Any:
        if hasattr(self.env, "model_copy"):
            return self.env.model_copy(deep=True)
        if hasattr(self.env, "copy"):
            try:
                return self.env.copy(deep=True)
            except TypeError:
                return self.env.copy()
        raise TypeError("Shadow execution requires an environment copy method.")

    @staticmethod
    def _snapshot(env: Any) -> dict:
        if hasattr(env, "model_dump"):
            return env.model_dump(mode="json")
        if hasattr(env, "dict"):
            return env.dict()
        return {}

    @staticmethod
    def _state_diff(before: dict, after: dict) -> str:
        try:
            from deepdiff import DeepDiff

            diff = DeepDiff(before, after, ignore_order=True, verbose_level=2)
            return diff.to_json(default_mapping={object: str})[:4000]
        except Exception as exc:
            return json.dumps({"diff_error": str(exc)}, ensure_ascii=False)

    def shadow_execute(self, tool_name: str, tool_arguments: dict) -> ShadowExecutionResult:
        shadow_env = self._copy_env()
        before = self._snapshot(self.env)
        result, error = self.runtime.run_function(shadow_env, tool_name, tool_arguments)
        after = self._snapshot(shadow_env)
        return ShadowExecutionResult(
            result=self._format(result),
            error=error,
            state_diff=self._state_diff(before, after),
        )

    def commit(self, tool_name: str, tool_arguments: dict) -> tuple[str, str | None]:
        result, error = self.runtime.run_function(self.env, tool_name, tool_arguments)
        return self._format(result), error


class TransactionRollbackDBExecutor:
    """Shadow executor for database tools using transaction rollback.

    The supplied ``query_runner`` is called with ``(connection, tool_name,
    tool_arguments)`` and must execute the database operation using that
    connection. For shadow execution, SimGuard opens a transaction, runs the
    operation, records the concrete result, and rolls the transaction back. If
    the call is allowed, ``commit`` reruns the same operation against the live
    connection and commits it.

    Works with SQLAlchemy Engine/Connection and most Python DB-API connections.
    """

    def __init__(
        self,
        connectable: Any,
        query_runner: Callable[[Any, str, dict], Any],
        result_formatter: Callable[[Any], str] | None = None,
        diff_runner: Callable[[Any], Any] | None = None,
    ) -> None:
        self.connectable = connectable
        self.query_runner = query_runner
        self.result_formatter = result_formatter or self._default_format
        self.diff_runner = diff_runner

    @staticmethod
    def _default_format(result: Any) -> str:
        if result is None:
            return "null"
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except TypeError:
            return str(result)

    def _format(self, result: Any) -> str:
        try:
            return self.result_formatter(result)
        except Exception:
            return str(result)

    def _connect(self) -> tuple[Any, bool]:
        if hasattr(self.connectable, "connect"):
            return self.connectable.connect(), True
        return self.connectable, False

    def _begin(self, conn: Any) -> Any:
        if hasattr(conn, "begin"):
            return conn.begin()
        cursor = conn.cursor()
        cursor.execute("BEGIN")
        return None

    @staticmethod
    def _rollback(conn: Any, tx: Any) -> None:
        if tx is not None and hasattr(tx, "rollback"):
            tx.rollback()
        elif hasattr(conn, "rollback"):
            conn.rollback()

    @staticmethod
    def _commit(conn: Any, tx: Any) -> None:
        if tx is not None and hasattr(tx, "commit"):
            tx.commit()
        elif hasattr(conn, "commit"):
            conn.commit()

    @staticmethod
    def _close(conn: Any, should_close: bool) -> None:
        if should_close and hasattr(conn, "close"):
            conn.close()

    def _snapshot(self, conn: Any) -> Any:
        if self.diff_runner is None:
            return None
        return self.diff_runner(conn)

    @staticmethod
    def _diff(before: Any, after: Any) -> str:
        if before is None and after is None:
            return "transaction_rolled_back"
        try:
            from deepdiff import DeepDiff

            return DeepDiff(before, after, ignore_order=True, verbose_level=2).to_json(default_mapping={object: str})[:4000]
        except Exception as exc:
            return json.dumps({"diff_error": str(exc)}, ensure_ascii=False)

    def shadow_execute(self, tool_name: str, tool_arguments: dict) -> ShadowExecutionResult:
        conn, should_close = self._connect()
        tx = None
        before = None
        after = None
        try:
            tx = self._begin(conn)
            before = self._snapshot(conn)
            result = self.query_runner(conn, tool_name, tool_arguments)
            after = self._snapshot(conn)
            return ShadowExecutionResult(
                result=self._format(result),
                error=None,
                state_diff=self._diff(before, after),
                metadata={"shadow_mode": "transaction_rollback"},
            )
        except Exception as exc:
            return ShadowExecutionResult(
                result="",
                error=f"{type(exc).__name__}: {exc}",
                state_diff=self._diff(before, after),
                metadata={"shadow_mode": "transaction_rollback"},
            )
        finally:
            if tx is not None or hasattr(conn, "rollback"):
                self._rollback(conn, tx)
            self._close(conn, should_close)

    def commit(self, tool_name: str, tool_arguments: dict) -> tuple[str, str | None]:
        conn, should_close = self._connect()
        tx = None
        try:
            tx = self._begin(conn)
            result = self.query_runner(conn, tool_name, tool_arguments)
            self._commit(conn, tx)
            return self._format(result), None
        except Exception as exc:
            if tx is not None or hasattr(conn, "rollback"):
                self._rollback(conn, tx)
            return "", f"{type(exc).__name__}: {exc}"
        finally:
            self._close(conn, should_close)
