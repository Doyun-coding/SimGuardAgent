"""
DatabaseExecutor

DB 도구 호출을 실제 DB의 트랜잭션 안에서 실행해 결과와 상태 변화를 본 뒤 되돌린다(rollback).
ToolCallSimulator는 이 결과를 LLM 예상 대신 도구 결과로 쓴다. 실제 DB에는 아무것도 남지 않는다.

  DatabaseExecutor("app.db")                          SQLite 파일
  DatabaseExecutor("postgresql://user@host/db")       SQLAlchemy URL (SQLAlchemy 필요)
  DatabaseExecutor(engine)                            SQLAlchemy Engine
  DatabaseExecutor(lambda: psycopg.connect(...))      DB-API 연결을 만드는 함수
  DatabaseExecutor("app.db", copy=True)               실제 DB를 판정 순간의 상태로 복사해 복사본에서 실행 (SQLite)

보통은 protect(database=...)로 넘긴다. 그러면 SQL이 든 도구 호출을 이 실행기로 판정한다.

도구 인자에 SQL 문(query, sql, statement 등)이 있으면 그대로 실행한다. 도구가 SQL을 직접 받지
않으면 run(connection, tool_name, arguments)로 그 도구가 하는 DB 작업을 넘긴다.

되돌릴 수 없는 문장은 실행하지 않는다
  MySQL·MariaDB·Oracle은 CREATE, ALTER, DROP, TRUNCATE, RENAME을 실행하면 트랜잭션과 상관없이 바로
  확정한다. 실행하면 실제 DB가 바뀌므로, 이런 문장은 실행하지 않고 None을 돌려준다.
  그러면 ToolCallSimulator가 실행기가 없을 때처럼 내부 모델로 결과를 예상해 판정한다.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

from simguard.live_api import ToolRunResult

_SQL_KEYS = ("query", "sql", "statement", "sql_query", "command")
_MAX_ROWS = 50
_IMPLICIT_COMMIT_DIALECTS = {"mysql", "mariadb", "oracle"}
_IMPLICIT_COMMIT_SQL = re.compile(r"^\s*(create|alter|drop|truncate|rename)\b", re.IGNORECASE)
_SQL = re.compile(
    r"^\s*(select|insert|update|delete|drop|create|alter|truncate|replace|with|pragma|attach|vacuum|merge|grant|revoke)\b",
    re.IGNORECASE,
)


def is_sql_call(arguments: Any) -> bool:
    """도구 인자 값 중에 SQL 문이 있는가."""
    if not isinstance(arguments, Mapping):
        return False
    return any(isinstance(value, str) and _SQL.match(value) for value in arguments.values())


class DatabaseExecutor:
    def __init__(
        self,
        database: Any,
        run: Callable[[Any, str, dict], Any] | None = None,
        snapshot: Callable[[Any], Any] | None = None,
        copy: bool = False,
    ) -> None:
        """
        database: SQLite 파일 경로, SQLAlchemy URL이나 Engine, 또는 DB-API 연결을 만드는 함수
        run:      도구가 SQL을 직접 받지 않을 때, 주어진 연결로 그 도구의 DB 작업을 하는 함수
        snapshot: 실행 전후 상태를 비교할 값을 만드는 함수. 없으면 스키마와 바뀐 행 수를 비교한다.
        copy:     True면 SQLite DB를 판정할 때마다 그 순간의 상태로 임시 파일에 복사하고 복사본에서
                  실행한다 (경로로 받았으면 읽기 전용으로 열고, 연결 함수로 받았으면 그 연결에서 복사한다).
                  SQLite가 아닌 DB는 복사하지 않고 트랜잭션 안에서 실행하고 되돌린다.
        """
        self._database = database
        self._run = run
        self._snapshot = snapshot
        self._engine = None
        self._copy = copy
        self._copies: dict[int, str] = {}

    # ── 연결 ──────────────────────────────────────────────────────────────

    def _sqlite_copy(self, source: sqlite3.Connection) -> tuple[Any, str, str]:
        """판정 순간의 DB를 임시 파일로 복사하고 복사본 연결을 돌려준다. source는 닫는다.
        backup()은 다른 연결이 쓰는 중이어도 일관된 상태를 복사한다."""
        fd, copy_path = tempfile.mkstemp(prefix="simguard_live_", suffix=".sqlite")
        os.close(fd)
        try:
            target = sqlite3.connect(copy_path, check_same_thread=False)
            source.backup(target)
        finally:
            source.close()
        self._copies[id(target)] = copy_path
        return self._sqlite(target)

    def _sqlite_path(self, path: str) -> tuple[Any, str, str]:
        if not self._copy:
            return self._sqlite(sqlite3.connect(path, check_same_thread=False))
        # 실제 DB 파일은 읽기 전용으로만 연다.
        return self._sqlite_copy(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True,
                                                 check_same_thread=False))

    @staticmethod
    def _sqlite(conn: sqlite3.Connection) -> tuple[Any, str, str]:
        # 파이썬 sqlite3의 기본 설정은 DDL(DROP, ALTER 등)을 트랜잭션 밖에서 실행해 바로 확정한다.
        # 트랜잭션을 직접 열면 SQLite는 DDL도 되돌린다.
        if conn.in_transaction:
            conn.rollback()
        conn.isolation_level = None
        conn.execute("BEGIN")
        return conn, "sqlite", "sqlite"

    def _open(self) -> tuple[Any, str, str]:
        """(연결, 종류, 방언 이름). 종류: sqlite | sqlalchemy | dbapi"""
        database = self._database
        if isinstance(database, str) and "://" not in database:
            return self._sqlite_path(database)
        if isinstance(database, str) or hasattr(database, "dialect"):
            if self._engine is None:
                if isinstance(database, str):
                    from sqlalchemy import create_engine

                    self._engine = create_engine(database)
                else:
                    self._engine = database
            if self._engine.dialect.name == "sqlite" and self._engine.url.database not in (None, "", ":memory:"):
                # SQLAlchemy의 SQLite 드라이버도 같은 문제가 있어, 같은 파일을 직접 연다.
                return self._sqlite_path(self._engine.url.database)
            conn = self._engine.connect()
            conn.begin()
            return conn, "sqlalchemy", self._engine.dialect.name
        if callable(database):
            conn = database()
            if isinstance(conn, sqlite3.Connection):
                return self._sqlite_copy(conn) if self._copy else self._sqlite(conn)
            if getattr(conn, "autocommit", False) is True:
                conn.autocommit = False
            module = type(conn).__module__.lower()
            dialect = next((name for name in _IMPLICIT_COMMIT_DIALECTS if name in module), module.split(".")[0])
            return conn, "dbapi", dialect
        raise TypeError("database must be a SQLite path, a SQLAlchemy URL or Engine, or a function returning a DB-API connection")

    def _rollback_and_close(self, conn: Any, kind: str) -> None:
        try:
            if kind == "sqlite":
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            else:
                conn.rollback()
        finally:
            conn.close()
            copy_path = self._copies.pop(id(conn), None)
            if copy_path is not None:
                try:
                    os.unlink(copy_path)
                except OSError:
                    pass

    # ── 실행 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sql(arguments: Mapping[str, Any]) -> str | None:
        for key in _SQL_KEYS:
            if isinstance(arguments.get(key), str) and arguments[key].strip():
                return arguments[key]
        strings = [value for value in arguments.values() if isinstance(value, str) and value.strip()]
        return strings[0] if len(strings) == 1 else None

    @staticmethod
    def _execute_sql(conn: Any, kind: str, sql: str) -> dict:
        if kind == "sqlalchemy":
            result = conn.exec_driver_sql(sql)
            if result.returns_rows:
                rows = result.fetchmany(_MAX_ROWS + 1)
                return {"columns": list(result.keys()), "rows": [list(row) for row in rows[:_MAX_ROWS]],
                        "truncated": len(rows) > _MAX_ROWS}
            return {"rows_affected": result.rowcount}
        cursor = conn.cursor()
        cursor.execute(sql)
        if cursor.description:
            rows = cursor.fetchmany(_MAX_ROWS + 1)
            return {"columns": [column[0] for column in cursor.description],
                    "rows": [list(row) for row in rows[:_MAX_ROWS]], "truncated": len(rows) > _MAX_ROWS}
        return {"rows_affected": cursor.rowcount}

    @staticmethod
    def _schema(conn: Any, kind: str) -> dict:
        """테이블 이름과 정의."""
        try:
            if kind == "sqlite":
                rows = conn.execute("SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
                return {f"{kind_}:{name}": sql for kind_, name, sql in rows}
            if kind == "sqlalchemy":
                from sqlalchemy import inspect

                inspector = inspect(conn)
                return {
                    f"table:{name}": [f"{column['name']} {column['type']}" for column in inspector.get_columns(name)]
                    for name in inspector.get_table_names()
                }
        except Exception as exc:
            return {"schema_error": str(exc)}
        return {}

    def _state(self, conn: Any, kind: str) -> Any:
        return self._snapshot(conn) if self._snapshot is not None else self._schema(conn, kind)

    @staticmethod
    def _diff(before: Any, after: Any) -> dict:
        if isinstance(before, dict) and isinstance(after, dict):
            return {
                "added": {key: after[key] for key in after.keys() - before.keys()},
                "removed": sorted(before.keys() - after.keys()),
                "changed": {key: after[key] for key in before.keys() & after.keys() if before[key] != after[key]},
            }
        return {} if before == after else {"before": before, "after": after}

    def live_execute(self, tool_name: str, tool_arguments: dict) -> ToolRunResult:
        arguments = dict(tool_arguments or {})
        sql = None if self._run is not None else self._sql(arguments)
        if self._run is None and sql is None:
            return ToolRunResult(
                result="", error="DatabaseExecutor: no SQL statement in the arguments and no run function",
                metadata={"live_mode": "transaction_rollback", "executed": False},
            )
        conn, kind, dialect = self._open()
        mode = "copy" if id(conn) in self._copies else "transaction_rollback"
        metadata = {"live_mode": mode, "dialect": dialect, "executed": False}
        try:
            if sql is not None and dialect in _IMPLICIT_COMMIT_DIALECTS and _IMPLICIT_COMMIT_SQL.match(sql):
                # 실행하면 실제 DB에 바로 확정되므로 실행하지 않는다. None을 돌려주면 ToolCallSimulator가
                # 실행하지 않고 내부 모델로 결과를 예상한다.
                return None
            before = self._state(conn, kind)
            try:
                result = self._run(conn, tool_name, arguments) if self._run is not None else self._execute_sql(conn, kind, sql)
                error = None
            except Exception as exc:
                result, error = "", f"{type(exc).__name__}: {exc}"
            metadata["executed"] = True
            after = self._state(conn, kind)
            diff = self._diff(before, after)
            if isinstance(result, dict) and "rows_affected" in result:
                diff = {"rows_affected": result["rows_affected"], **diff}
            return ToolRunResult(
                result=result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str),
                error=error,
                state_diff=json.dumps(diff, ensure_ascii=False, default=str)[:4000],
                metadata=metadata,
            )
        finally:
            self._rollback_and_close(conn, kind)
