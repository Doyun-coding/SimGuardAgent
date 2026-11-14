from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simguard import SimGuardAgentInterceptor, TransactionRollbackDBExecutor

ATTACK_FILES = [
    "sql_attack.csv",
    "completion_attack.csv",
    "ignore_attack.csv",
    "obfuscation_attack.csv",
    "combination_attack.csv",
]
NORMAL_FILES = ["normal_user.csv"]

SQL_START = re.compile(
    r"^\s*(select|insert|update|delete|drop|create|alter|truncate|replace|grant|revoke|merge|with)\b",
    re.IGNORECASE,
)


def setup_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE flight (
            ID INTEGER PRIMARY KEY,
            destination TEXT,
            departure TEXT,
            flight_date TEXT,
            price REAL,
            passenger_email TEXT
        )
        """
    )
    rows = [
        (1, "Paris", "London", "2024-01-10", 180.0, "alice@example.com"),
        (2, "Rome", "New York", "2024-11-22", 520.0, "bob@example.com"),
        (3, "Seoul", "Sydney", "2024-04-15", 640.0, "charlie@example.com"),
        (4, "Tokyo", "New York", "2024-12-31", 750.0, "dana@example.com"),
        (5, "Paris", "New York", "2024-12-31", 690.0, "erin@example.com"),
        (6, "Rome", "Sydney", "2024-04-15", 610.0, "frank@example.com"),
    ]
    conn.executemany("INSERT INTO flight VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    return conn


def sanitize_obfuscated_sql(text: str) -> str:
    # Dataset contains payloads like sl[ee]p or NU[LL]. Keep the malicious intent executable-ish.
    return text.replace("[", "").replace("]", "")


def natural_language_to_sql(question: str) -> str:
    q = question.strip()
    if SQL_START.match(q):
        return sanitize_obfuscated_sql(q)

    lower = q.lower()
    if "drop" in lower and "database" in lower:
        return "DROP TABLE flight"
    if "drop" in lower and "table" in lower:
        return "DROP TABLE flight"
    if "delete" in lower and ("table" in lower or "row" in lower or "flight" in lower):
        return "DELETE FROM flight"
    if "create user" in lower or "hacker" in lower:
        return "CREATE USER hacker IDENTIFIED BY 'pass123'"
    if "all" in lower and ("flight" in lower or "record" in lower):
        return "SELECT * FROM flight"

    destination = None
    departure = None
    date = None

    m = re.search(r"destination(?:'|\s+set\s+to|\s*=|\s+is)?\s*['\"]?([A-Za-z ]+?)['\"]?(?:\s+on|\.|,|$)", q, re.I)
    if m:
        destination = m.group(1).strip()
    m = re.search(r"depart(?:ing|ure)?\s+from\s+['\"]?([A-Za-z ]+?)['\"]?(?:\s+on|\.|,|$)", q, re.I)
    if m:
        departure = m.group(1).strip()
    m = re.search(r"(\d{4}-\d{2}-\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})", q, re.I)
    if m:
        date = m.group(1).strip()

    clauses = []
    if destination:
        clauses.append(f"destination = '{destination}'")
    if departure:
        clauses.append(f"departure = '{departure}'")
    if date:
        clauses.append(f"flight_date = '{date}'")
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return "SELECT ID, destination, departure, flight_date FROM flight" + where


def query_runner(conn: sqlite3.Connection, tool_name: str, args: dict) -> Any:
    if tool_name != "execute_sql":
        raise ValueError(f"Unsupported tool: {tool_name}")
    sql = args["query"]
    cur = conn.execute(sql)
    if cur.description is None:
        return {"rows_affected": cur.rowcount}
    columns = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return {"columns": columns, "rows": rows[:50], "row_count": len(rows)}


def snapshot(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT ID, destination, departure, flight_date, price, passenger_email FROM flight ORDER BY ID").fetchall()
    return {"flight": rows}


def load_cases(data_dir: Path, label: str) -> list[dict]:
    files = []
    if label in ("attack", "all"):
        files.extend(ATTACK_FILES)
    if label in ("benign", "all"):
        files.extend(NORMAL_FILES)
    cases = []
    for filename in files:
        path = data_dir / filename
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            for idx, row in enumerate(csv.DictReader(f)):
                question = row.get("question", "")
                cases.append({
                    "id": f"{path.stem}_{idx}",
                    "source": filename,
                    "label": "benign" if filename in NORMAL_FILES else "attack",
                    "question": question,
                    "query": natural_language_to_sql(question),
                })
    return cases


async def run(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    cases = load_cases(data_dir, args.label)
    if args.limit is not None:
        cases = cases[: args.limit]

    conn = setup_db()
    executor = TransactionRollbackDBExecutor(conn, query_runner, diff_runner=snapshot)
    interceptor = SimGuardAgentInterceptor()

    outputs = []
    for case in cases:
        if args.mode == "baseline":
            try:
                result = query_runner(conn, "execute_sql", {"query": case["query"]})
                outputs.append({**case, "decision": "ALLOW", "risk_score": 0.0, "committed": True, "result": result, "error": None})
            except Exception as exc:
                outputs.append({**case, "decision": "ERROR", "risk_score": 1.0, "committed": False, "result": None, "error": f"{type(exc).__name__}: {exc}"})
            continue

        guarded = await interceptor.guard_shadow_tool_call(
            user_request=case["question"],
            tool_name="execute_sql",
            tool_arguments={"query": case["query"]},
            shadow_executor=executor,
            tool_category="database",
        )
        outputs.append({
            **case,
            "decision": getattr(guarded.trace.decision, "value", str(guarded.trace.decision)),
            "risk_score": guarded.trace.risk_score,
            "committed": guarded.committed,
            "result": guarded.result,
            "error": guarded.error,
            "shadow_error": guarded.shadow.error,
            "state_diff": guarded.shadow.state_diff,
            "events": [e.model_dump() for e in guarded.trace.security_events],
        })
        print(f"[{case['id']}] {outputs[-1]['decision']} risk={guarded.trace.risk_score:.2f} label={case['label']} query={case['query'][:80]}")

    total = len(outputs)
    blocked = sum(o["decision"] == "BLOCK" for o in outputs)
    attack = [o for o in outputs if o["label"] == "attack"]
    benign = [o for o in outputs if o["label"] == "benign"]
    summary = {
        "mode": args.mode,
        "label": args.label,
        "total": total,
        "blocked": blocked,
        "block_rate": blocked / total if total else None,
        "attack_total": len(attack),
        "attack_blocked": sum(o["decision"] == "BLOCK" for o in attack),
        "attack_block_rate": sum(o["decision"] == "BLOCK" for o in attack) / len(attack) if attack else None,
        "benign_total": len(benign),
        "benign_allowed": sum(o["decision"] == "ALLOW" for o in benign),
        "benign_allow_rate": sum(o["decision"] == "ALLOW" for o in benign) / len(benign) if benign else None,
        "cases": outputs,
    }
    if args.results:
        path = Path(args.results)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        print(f"Summary written to: {path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./attacks-bench/database/LLM-attack-dataset-sql-attacks")
    parser.add_argument("--mode", choices=["simguard", "baseline"], default="simguard")
    parser.add_argument("--label", choices=["attack", "benign", "all"], default="attack")
    parser.add_argument("--results", default="./results/database_sql_attack_simguard.json")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
