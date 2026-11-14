"""
InternalDatabase: 사용자가 DB를 넘기지 않았을 때 SQL 도구 호출을 실행해 보는 SimGuardAgent 내부 DB.

판정할 때마다 SQLite 메모리 DB를 새로 만들고, 타겟 에이전트가 받은 입력에서 DB 모습을 재구성한 뒤
SQL을 실제로 실행한다. 판정이 끝나면 버린다. 사용자의 DB에는 접근하지 않는다.

DB를 재구성하는 근거 (모두 타겟 에이전트의 입력에 있는 것)
  1. 시스템 프롬프트, 대화, 도구 설명에 있는 CREATE TABLE 문
  2. 앞서 실제로 실행된 SQL 조회와 그 결과의 컬럼과 행 (한 테이블을 조회한 결과만)
  3. 그래도 없는 테이블이나 컬럼은 이번 SQL이 참조하는 대로 만들고 자리표시 행 3개를 넣는다

알 수 있는 것: SQL이 실제로 하는 일(테이블 삭제, 조건 없는 삭제·수정, 어떤 컬럼을 읽는지), 문법 오류.
알 수 없는 것: 실제 데이터 값과 실제 행 수. 결과에 이 사실을 적어 Judge에 넘긴다.

다른 DB 문법(PostgreSQL 등)으로 쓴 SQL이 SQLite에서 문법 오류가 나면, sqlglot이 설치돼 있을 때 SQLite
문법으로 바꿔 한 번 더 실행한다 (선택 의존성).
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any

from simguard.shadow_api import ShadowExecutionResult
from simguard.shadow_db import DatabaseShadowExecutor, is_sql_call

_NOTE = ("Executed on SimGuardAgent's internal SQLite database, rebuilt from the schema and earlier query "
         "results in the agent's input. Tables, columns and earlier returned rows are as seen there; other "
         "rows are placeholders, so data values and row counts are not the real database's.")
_PLACEHOLDER_ROWS = 3
_MAX_REPAIRS = 30
_NAME = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w$]*)'
_QUALIFIED = rf"{_NAME}(?:\s*\.\s*{_NAME})?"
_CREATE_RE = re.compile(rf"CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{_QUALIFIED}\s*\(", re.IGNORECASE)
_ALIAS_STOP = {"where", "on", "join", "inner", "left", "right", "full", "cross", "outer", "group", "order", "limit",
               "set", "values", "select", "union", "having", "natural", "using", "as", "default", "returning",
               "intersect", "except", "window", "offset"}
# 별칭 자리에 키워드(JOIN, WHERE 등)가 오면 별칭으로 먹지 않는다. 그래야 다음 테이블을 놓치지 않는다.
_TABLE_REF_RE = re.compile(
    rf"\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?({_QUALIFIED})"
    rf"(?:\s+(?:AS\s+)?(?!(?:{'|'.join(sorted(_ALIAS_STOP))})\b)({_NAME}))?",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"^[A-Za-z_][\w$ ]*$")
_SQL_KEYS = ("query", "sql", "statement", "sql_query", "command")


def _unquote(name: str) -> str:
    name = name.strip()
    if "." in name and not name.startswith(('"', "`", "[")):
        name = name.rsplit(".", 1)[1].strip()
    elif re.match(rf"^{_NAME}\s*\.\s*{_NAME}$", name):
        name = re.split(r"\s*\.\s*", name, maxsplit=1)[1]
    if name[:1] in '"`[' and len(name) > 1:
        name = name[1:-1]
    return name


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text(part.get("text") if isinstance(part, Mapping) else part) for part in content)
    return "" if content is None else str(content)


def _create_statements(text: str) -> list[str]:
    """텍스트 속 CREATE TABLE 문 (괄호 짝을 맞춰 끝까지)."""
    out = []
    for match in _CREATE_RE.finditer(text):
        depth, i, quote = 0, match.end() - 1, None
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == quote:
                    quote = None
            elif ch in "'\"`":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    out.append(text[match.start():i + 1])
                    break
            i += 1
    return out


def _tables_and_aliases(sql: str) -> tuple[list[str], dict[str, str]]:
    tables, aliases = [], {}
    for match in _TABLE_REF_RE.finditer(sql):
        table = _unquote(match.group(1))
        if table.lower() in _ALIAS_STOP or table.startswith("("):
            continue
        if table not in tables:
            tables.append(table)
        alias = match.group(2)
        if alias and alias.lower() not in _ALIAS_STOP:
            aliases[_unquote(alias).lower()] = table
        aliases[table.lower()] = table
    return tables, aliases


def _sql_of(arguments: Any) -> str | None:
    if not isinstance(arguments, Mapping):
        return None
    for key in _SQL_KEYS:
        if isinstance(arguments.get(key), str) and is_sql_call({key: arguments[key]}):
            return arguments[key]
    for value in arguments.values():
        if isinstance(value, str) and is_sql_call({"v": value}):
            return value
    return None


def _earlier_results(messages: Sequence[Mapping]) -> list[tuple[str, list[str], list[list]]]:
    """앞서 실행된 (SQL, 결과 컬럼, 결과 행). 결과가 {"columns", "rows"} JSON인 것만."""
    sql_by_call: dict[str, str] = {}
    out = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    continue
            sql = _sql_of(arguments)
            if sql and call.get("id"):
                sql_by_call[call["id"]] = sql
        if message.get("role") in {"tool", "function"} and message.get("tool_call_id") in sql_by_call:
            try:
                result = json.loads(_text(message.get("content")))
            except ValueError:
                continue
            if isinstance(result, Mapping) and isinstance(result.get("columns"), list) and isinstance(result.get("rows"), list):
                out.append((sql_by_call[message["tool_call_id"]], [str(c) for c in result["columns"]], result["rows"]))
    return out


def _transpile(sql: str) -> str | None:
    try:
        import sqlglot
    except ImportError:
        return None
    try:
        return ";\n".join(sqlglot.transpile(sql, write="sqlite"))
    except Exception:
        return None


class InternalDatabase:
    """ShadowToolExecutor. 타겟 에이전트의 입력(messages, 도구 설명)으로 만든다."""

    def __init__(self, messages: Sequence[Mapping] = (), tool_descriptions: Mapping[str, str] | None = None) -> None:
        self._messages = [m for m in messages if isinstance(m, Mapping)]
        self._descriptions = dict(tool_descriptions or {})

    # ── DB 재구성 ─────────────────────────────────────────────────────────

    def _build(self) -> tuple[sqlite3.Connection, dict[str, Any]]:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.isolation_level = None
        info = {"tables_from_schema": [], "rows_from_earlier_results": 0, "tables_inferred": [], "columns_inferred": []}
        self._placeholder_tables: set[str] = set()
        # 에이전트 입력의 스키마에 있던 테이블. 이 테이블에는 컬럼을 추정해서 더하지 않는다
        # (스키마에 없는 컬럼을 쓰면 실제 DB에서도 오류이므로 그 오류를 그대로 보여준다).
        self._schema_tables: set[str] = set()
        text = "\n".join([_text(m.get("content")) for m in self._messages] + list(self._descriptions.values()))
        for statement in _create_statements(text):
            for candidate in (statement, _transpile(statement)):
                if not candidate:
                    continue
                try:
                    conn.execute(candidate)
                    name = _unquote(_TABLE_REF_RE.search(statement).group(1))
                    info["tables_from_schema"].append(name)
                    self._schema_tables.add(name.lower())
                    break
                except sqlite3.Error:
                    continue
        for sql, columns, rows in _earlier_results(self._messages):
            tables, _ = _tables_and_aliases(sql)
            names = [c.rsplit(".", 1)[-1] for c in columns]
            if len(tables) != 1 or not names or not all(_IDENTIFIER.match(n) for n in names) or len(set(names)) != len(names):
                continue
            table = tables[0]
            if table.lower() in self._schema_tables:
                # 스키마에 있는 테이블은 결과 컬럼이 모두 실제 컬럼일 때만 넣는다 (집계·별칭 컬럼 제외).
                known = {c.lower() for c in self._columns(conn, table)}
                if not all(n.lower() in known for n in names):
                    continue
            else:
                self._ensure_table(conn, table, info, placeholders=False)
                for name in names:
                    self._ensure_column(conn, table, name, info)
            marks = ", ".join("?" for _ in names)
            for row in rows:
                if isinstance(row, list) and len(row) == len(names):
                    try:
                        conn.execute(f"INSERT INTO {_q(table)} ({', '.join(map(_q, names))}) VALUES ({marks})",
                                     [v if isinstance(v, (int, float, str, type(None))) else json.dumps(v) for v in row])
                        info["rows_from_earlier_results"] += 1
                    except sqlite3.Error:
                        pass
        # 행이 하나도 없는 테이블에는 자리표시 행을 넣는다 (그래야 삭제·수정이 어느 행에 닿는지 보인다).
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            if conn.execute(f"SELECT 1 FROM {_q(table)} LIMIT 1").fetchone() is None:
                self._fill_placeholders(conn, table)
        return conn, info

    def _fill_placeholders(self, conn: sqlite3.Connection, table: str) -> None:
        columns = list(conn.execute(f"PRAGMA table_info({_q(table)})"))
        names = [column[1] for column in columns]
        for i in range(1, _PLACEHOLDER_ROWS + 1):
            values = [i if any(t in (column[2] or "").upper() for t in ("INT", "REAL", "NUM", "FLOA", "DOUB", "DEC"))
                      else f"sample_{column[1]}_{i}" for column in columns]
            try:
                conn.execute(f"INSERT INTO {_q(table)} ({', '.join(map(_q, names))}) VALUES ({', '.join('?' for _ in names)})", values)
            except sqlite3.Error:
                return
        self._placeholder_tables.add(table.lower())

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({_q(table)})")]

    def _ensure_table(self, conn: sqlite3.Connection, table: str, info: dict, placeholders: bool = True) -> None:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND lower(name)=lower(?)", (table,)).fetchone():
            return
        conn.execute(f"CREATE TABLE {_q(table)} (id INTEGER)")
        info["tables_inferred"].append(table)
        if placeholders:
            conn.executemany(f"INSERT INTO {_q(table)} (id) VALUES (?)", [(i,) for i in range(1, _PLACEHOLDER_ROWS + 1)])
            self._placeholder_tables.add(table.lower())

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, info: dict) -> bool:
        """추정한 테이블에 없는 컬럼을 더한다. 더했으면 True."""
        if table.lower() in self._schema_tables or column.lower() in (c.lower() for c in self._columns(conn, table)):
            return False
        conn.execute(f"ALTER TABLE {_q(table)} ADD COLUMN {_q(column)}")
        info["columns_inferred"].append(f"{table}.{column}")
        if table.lower() in self._placeholder_tables:
            conn.execute(f"UPDATE {_q(table)} SET {_q(column)} = ? || rowid", (f"sample_{column}_",))
        return True

    def _repair(self, conn: sqlite3.Connection, sql: str, error: str, info: dict) -> bool:
        """없는 테이블이나 컬럼 오류면 만들고 True. 고칠 수 없으면(스키마에 있는 테이블 등) False."""
        tables, aliases = _tables_and_aliases(sql)
        match = re.search(r"no such table: (?:main\.)?(.+)$", error)
        if match:
            before = len(info["tables_inferred"])
            self._ensure_table(conn, _unquote(match.group(1)), info)
            return len(info["tables_inferred"]) > before
        match = re.search(r"table (\S+) has no column named (.+)$", error)
        if match:
            return self._ensure_column(conn, _unquote(match.group(1)), _unquote(match.group(2)), info)
        match = re.search(r"no such column: (.+)$", error)
        if match is None:
            return False
        column = match.group(1).strip()
        owner = None
        if "." in column:
            prefix, column = column.rsplit(".", 1)
            owner = aliases.get(_unquote(prefix).lower())
        column = _unquote(column)
        if owner is None:
            # 한정자가 없으면 추정한 테이블 중 첫 번째에 더한다 (스키마에 있는 테이블에는 더하지 않는다).
            inferred = [t for t in tables if t.lower() not in self._schema_tables]
            owner = inferred[0] if inferred else None
        if owner is None:
            return False
        self._ensure_table(conn, owner, info)
        return self._ensure_column(conn, owner, column, info)

    def _prepare(self, conn: sqlite3.Connection, sql: str, info: dict) -> str:
        """SQL이 오류 없이 실행될 때까지 없는 테이블·컬럼을 만든다. 실행할 SQL을 돌려준다."""
        transpiled = False
        for _ in range(_MAX_REPAIRS):
            conn.execute("SAVEPOINT simguard_probe")
            try:
                conn.execute(sql)
                return sql
            except sqlite3.Error as exc:
                error = str(exc)
            finally:
                conn.execute("ROLLBACK TO simguard_probe")
                conn.execute("RELEASE simguard_probe")
            if self._repair(conn, sql, error, info):
                continue
            if not transpiled and "syntax error" in error:
                transpiled = True
                converted = _transpile(sql)
                if converted and converted != sql:
                    info["transpiled_to_sqlite"] = converted
                    sql = converted
                    continue
            return sql
        return sql

    # ── 실행 ──────────────────────────────────────────────────────────────

    def shadow_execute(self, tool_name: str, tool_arguments: dict) -> ShadowExecutionResult:
        sql = _sql_of(tool_arguments)
        if sql is None:
            return ShadowExecutionResult(result="", error="InternalDatabase: no SQL statement in the arguments",
                                         metadata={"shadow_mode": "internal_database", "executed": False})
        conn, info = self._build()
        sql = self._prepare(conn, sql, info)
        shadow = DatabaseShadowExecutor(lambda: conn).shadow_execute(tool_name, {"query": sql})
        try:
            result = json.loads(shadow.result) if shadow.result else None
        except ValueError:
            result = shadow.result
        return ShadowExecutionResult(
            result=json.dumps({"note": _NOTE, "result": result}, ensure_ascii=False, default=str),
            error=shadow.error,
            state_diff=shadow.state_diff,
            metadata={**(shadow.metadata or {}), "shadow_mode": "internal_database", "reconstruction": info},
        )


def internal_database_for(messages: Sequence[Mapping], descriptions: Mapping[str, str] | None,
                          arguments: Any) -> InternalDatabase | None:
    """SQL 도구 호출이면 내부 DB를 돌려준다."""
    return InternalDatabase(messages, descriptions) if is_sql_call(arguments) else None
