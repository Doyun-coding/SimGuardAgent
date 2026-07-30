"""OpenAPIMock (B) — OpenAPI 스펙에서 mock 응답을 자동 생성한다.

핵심: 엔드포인트를 손으로 짜지 않는다. 스펙에 정의된 경로/메서드/스키마를 읽어
  - 요청을 스펙의 operation에 매칭하고
  - requestBody를 스키마로 검증(위반 시 400)하고
  - 리소스형 경로(/users, /users/{id})는 제네릭 CRUD 상태로 처리해
    실제 부작용(삭제·생성·수정)을 응답에 드러낸다.

스펙의 x-mock-seed로 리소스 초기 상태를 준다. 민감 필드(비밀번호·시크릿)를 담아
유출이 응답으로 관찰되게 한다 — mock은 안전 필터를 걸지 않는다(결과를 보여주는 게
목적). 방어 판정은 judge가 한다.

스펙에 없는 요청은 처리하지 않고 None을 반환한다 → 라우터가 LLM mock(C)으로 넘긴다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft7Validator

_SPEC_DIR = Path(__file__).parent / "specs"


def load_specs(spec_dir: Path | None = None) -> list["OpenAPISpec"]:
    d = spec_dir or _SPEC_DIR
    return [OpenAPISpec(p) for p in sorted(d.glob("*.yaml"))] if d.exists() else []


class OpenAPISpec:
    """하나의 OpenAPI 문서. 경로 매칭과 스키마 검증을 담당한다."""

    def __init__(self, path: Path):
        self.doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.name = path.stem
        self.paths = self.doc.get("paths", {})
        self.seed = self.doc.get("x-mock-seed", {})
        # 경로 템플릿을 정규식으로 미리 컴파일
        self._compiled = []
        for tmpl in self.paths:
            regex = "^" + re.sub(r"\{[^}]+\}", r"[^/]+", tmpl) + "$"
            self._compiled.append((re.compile(regex), tmpl))

    def match(self, method: str, path: str) -> tuple[str, dict] | None:
        """(경로 템플릿, operation dict) 또는 None."""
        clean = path.split("?")[0]
        for rx, tmpl in self._compiled:
            if rx.match(clean):
                op = self.paths[tmpl].get(method.lower())
                if op is not None:
                    return tmpl, op
        return None


class OpenAPIMock:
    """스펙 묶음으로 요청을 처리한다. 제네릭 CRUD 상태 엔진 포함."""

    def __init__(self, specs: list[OpenAPISpec] | None = None):
        self.specs = specs if specs is not None else load_specs()

    def seed_state(self) -> dict[str, dict]:
        """모든 스펙의 x-mock-seed를 {collection: {id: item}}로 펼친다."""
        state: dict[str, dict] = {}
        for spec in self.specs:
            for coll, items in spec.seed.items():
                bucket = state.setdefault(coll, {})
                for it in items:
                    bucket[str(it.get("id"))] = dict(it)
        return state

    def handle(self, method: str, path: str, body: dict | None,
               state: dict[str, dict]) -> tuple[int, Any, list[str]] | None:
        """(status, body, side_effects) 또는 None(스펙에 없음 → C로).

        state는 호출 간 공유되어 부작용이 누적된다(stateful).
        """
        for spec in self.specs:
            m = spec.match(method, path)
            if m is None:
                continue
            tmpl, op = m
            # 1) requestBody 스키마 검증
            err = _validate_body(op, body)
            if err is not None:
                return 400, {"error": "schema validation failed", "detail": err}, []
            # 2) 제네릭 CRUD 처리
            return _crud(method, path, tmpl, body, state)
        return None


# ── 스키마 검증 ──────────────────────────────────────────────────────────────

def _validate_body(op: dict, body: dict | None) -> str | None:
    rb = op.get("requestBody")
    if not rb:
        return None
    schema = (rb.get("content", {}).get("application/json", {}).get("schema"))
    required = rb.get("required", False)
    if body is None:
        return "request body required" if required else None
    if not schema:
        return None
    errors = sorted(Draft7Validator(schema).iter_errors(body), key=lambda e: e.path)
    if errors:
        return errors[0].message
    return None


# ── 제네릭 CRUD 엔진 ─────────────────────────────────────────────────────────

def _collection_and_id(tmpl: str, path: str) -> tuple[str, str | None]:
    """/users → (users, None), /users/{id} → (users, 실제값), /secrets/{id} → (secrets, 값)."""
    clean = path.split("?")[0].strip("/")
    parts = clean.split("/")
    coll = parts[0] if parts else ""
    # 템플릿에 {…}가 있으면 마지막 세그먼트가 id
    has_id = "{" in tmpl
    item_id = parts[-1] if (has_id and len(parts) >= 2) else None
    return coll, item_id


def _crud(method: str, path: str, tmpl: str, body: dict | None,
          state: dict[str, dict]) -> tuple[int, Any, list[str]]:
    coll, item_id = _collection_and_id(tmpl, path)
    bucket = state.setdefault(coll, {})
    se: list[str] = []
    M = method.upper()

    # 컬렉션 단위
    if item_id is None:
        if M == "GET":
            items = list(bucket.values())
            # 민감 필드가 그대로 나가면 유출 관찰
            if any(_has_sensitive(it) for it in items):
                se.append(f"exposed {len(items)} {coll} records including sensitive fields")
            return 200, items, se
        if M == "POST":
            new_id = str(body.get("id") or (max((int(k) for k in bucket if k.isdigit()), default=0) + 1))
            item = {"id": new_id, **(body or {})}
            bucket[new_id] = item
            se.append(f"created {coll}/{new_id}")
            return 201, item, se
        if M == "DELETE":
            n = len(bucket); bucket.clear()
            se.append(f"DELETED ALL {n} {coll}")
            return 200, {"deleted_all": True, "removed": n}, se
        return 405, {"error": "method not allowed"}, se

    # 아이템 단위
    item = bucket.get(item_id)
    if M == "GET":
        if item is None:
            return 404, {"error": "not found"}, se
        if _has_sensitive(item):
            se.append(f"read sensitive {coll}/{item_id}")
        return 200, item, se
    if M in ("PUT", "PATCH"):
        if item is None:
            return 404, {"error": "not found"}, se
        before_role = item.get("role")
        item.update(body or {})
        if before_role and item.get("role") != before_role:
            se.append(f"changed role of {coll}/{item_id}: {before_role} → {item.get('role')}")
        else:
            se.append(f"modified {coll}/{item_id}")
        return 200, item, se
    if M == "DELETE":
        if item is None:
            return 404, {"error": "not found"}, se
        bucket.pop(item_id)
        se.append(f"deleted {coll}/{item_id}")
        return 200, {"deleted": item_id}, se
    return 405, {"error": "method not allowed"}, se


_SENSITIVE_KEYS = ("password", "secret", "value", "token", "key", "dsn", "salary", "ssn")


def _has_sensitive(item: dict) -> bool:
    if item.get("sensitive") is True:
        return True
    return any(any(s in k.lower() for s in _SENSITIVE_KEYS) for k in item)
