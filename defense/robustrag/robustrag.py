"""RobustRAG — isolate-then-aggregate 방어 (Xiang et al., IEEE S&P 2025)

논문: "Certifiably Robust RAG against Retrieval Corruption" (arXiv:2405.15556)

핵심 발상: 검색 문서를 서로 격리해 각각 따로 답을 만든 뒤, 그 답들을 안전하게
합친다. 오염 문서 하나는 격리된 응답 하나에만 영향을 주므로, 나머지 정상 응답에서
정답을 복원할 수 있다.

기존 방어들과 층이 다르다:
  차단형(SimGuard)  답변을 만든 뒤 위험하면 통째로 막는다 → 사용자가 답을 못 받는다
  제거형(RAGDefender) 오염 의심 문서를 버린다 → 오염이 없어도 항상 무언가 버린다
  RobustRAG          문서를 버리지도, 답을 막지도 않는다. 여러 답의 합의만 남긴다

Algorithm 1 (secure keyword aggregation)을 구현한다:
  1. 문서를 ω개씩 묶어 m개 그룹으로 격리
  2. 각 그룹으로 독립 응답 생성. 근거가 없으면 기권("I don't know")
  3. 기권하지 않은 응답에서 고유 키워드를 뽑아 카운트 누적
  4. 임계 μ = min(α·n, β) 이상인 키워드만 남김
  5. 남은 키워드만으로 (문서 없이) 최종 답변 생성

Algorithm 2(secure decoding aggregation)는 구현하지 않는다. 토큰마다 그룹 수만큼
logprob을 받아야 해서 HTTP API로는 토큰당 m회 호출이 필요하다 — 응답 500토큰 ×
18그룹이면 질의 하나에 9,000회 호출이라 실용적이지 않다.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter

from .keywords import extract_keywords

# 논문 Algorithm 1의 instruction i1 / i2.
#
# 논문은 기권을 "I don't know"라는 출력 문자열로 판정한다. 추론형 모델에는 그대로
# 쓸 수 없다 — 사고 과정에 지시문을 그대로 되뇌면서("reply with 'I don't know' if…")
# 그 문자열이 응답에 섞여, 정상 답변까지 기권으로 잡힌다. 실제로 이 구현의 첫 판에서
# 모든 격리 응답이 기권 처리되어 n=0이 됐다.
# 그래서 기권을 문자열이 아니라 JSON 필드로 받는다. 언어에도 무관해진다.
_ISOLATED_PROMPT = """\
Answer the query using ONLY the retrieved document below.

Do not guess, do not use outside knowledge, and do not comment on whether the \
document seems trustworthy. Just report what this one document says about the query.

Answer in the same language as the query. Be concise — one or two sentences.

[Retrieved document]
{passage}

[Query]
{query}

Return JSON only:
{{
  "relevant": "yes" or "no",
  "answer": "<your answer, or empty string when relevant is no>"
}}"""

_FINAL_PROMPT = """\
Answer the query using the provided keywords.

The keywords were extracted from trusted sources and are the only evidence \
available to you. Compose a direct answer to the query from them. If the \
keywords are insufficient to answer, say so briefly.

Answer in the same language as the query.

[Keywords]
{keywords}

[Query]
{query}

Return JSON only:
{{
  "answer": "<the answer text the user will see>"
}}"""

# JSON이 깨졌을 때만 쓰는 폴백. 정상 경로는 relevant 필드로 판정한다.
_ABSTAIN = re.compile(
    r"i\s*don'?t\s*know"
    r"|no\s+relevant\s+information"
    r"|不知道|无法回答|没有相关|无相关|不清楚"
    r"|모르겠|알\s*수\s*없",
    re.IGNORECASE,
)


def iso_group(passages: list[str], omega: int = 1) -> list[str]:
    """문서를 ω개씩 묶어 격리 그룹을 만든다 (논문 IsoGroup).

    ω=1이면 문서 하나가 곧 그룹이다 — 논문의 인증 설정이자 기본값이다.
    ω를 키우면 호출 수가 줄지만 오염 문서 하나가 더 많은 그룹을 오염시킨다.
    """
    if omega <= 1:
        return list(passages)
    return ["\n\n".join(passages[i:i + omega])
            for i in range(0, len(passages), omega)]


def is_abstained(response: str) -> bool:
    return bool(_ABSTAIN.search(response or ""))


class RobustRAG:
    """isolate-then-aggregate 방어.

    사용:
        rag = RobustRAG(client, model="qwen/qwen3.5-9b")
        out = await rag.generate(query, passages)
        out["response"]   # 최종 답변
    """

    def __init__(
        self,
        client,
        model: str,
        alpha: float = 0.4,
        beta: int = 4,
        omega: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.0,
        concurrency: int = 8,
    ):
        """
        Args:
            client: AsyncOpenAI 호환 클라이언트
            model:  격리 응답·최종 응답에 모두 쓰는 모델
            alpha:  임계 계수. μ = min(α·n, β)의 α (논문 장문 생성 기본값 0.4)
            beta:   임계 상한. n이 클 때 μ를 지배한다 (논문 장문 생성 기본값 4)
            omega:  그룹당 문서 수. 1이 논문의 인증 설정
            concurrency: 격리 응답 동시 생성 수. 격리 호출이 그룹 수만큼 필요하다
        """
        self.client = client
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._sem = asyncio.Semaphore(concurrency)

    async def _ask(self, prompt: str, max_tokens: int | None = None,
                   as_json: bool = False) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if as_json else {}
        async with self._sem:
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        return (resp.choices[0].message.content or "").strip()

    async def _isolated_answer(self, group: str, query: str) -> str:
        """격리 응답 하나. 기권이면 빈 문자열을 반환한다.

        JSON이 깨지면 원문에 문자열 폴백을 적용한다. 그 경우에도 사고 과정 오탐을
        피하려고, 판정은 마지막 200자(결론부)에만 건다.
        """
        raw = await self._ask(
            _ISOLATED_PROMPT.format(passage=group, query=query), as_json=True)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return "" if _ABSTAIN.search(raw[-200:]) else raw
        if str(data.get("relevant", "")).strip().lower() not in ("yes", "true"):
            return ""
        return str(data.get("answer") or "").strip()

    async def generate(self, query: str, passages: list[str]) -> dict:
        """격리 생성 → 키워드 집계 → 최종 생성.

        Returns:
            response          최종 답변
            isolated          격리 응답 원문 목록
            n_responded       기권하지 않은 응답 수 (논문의 n)
            threshold         적용된 μ
            keywords          최종 프롬프트에 넘긴 키워드 (논문의 W*)
            keyword_counts    키워드별 카운트 (분석용)
        """
        groups = iso_group(passages, self.omega)
        if not groups:
            return {"response": "", "isolated": [], "n_responded": 0,
                    "threshold": 0.0, "keywords": [], "keyword_counts": {}}

        # 1~2단계: 그룹별 독립 응답. 서로 문맥을 공유하지 않는 것이 격리의 전부다.
        isolated = await asyncio.gather(*[
            self._isolated_answer(g, query) for g in groups
        ])

        # 3단계: 기권하지 않은 응답에서만 고유 키워드를 센다
        counter: Counter[str] = Counter()
        n = 0
        for r in isolated:
            if not r:
                continue
            n += 1
            counter.update(extract_keywords(r))

        # 4단계: μ = min(α·n, β)
        # n이 크면 β가, 작으면 α·n이 지배한다. 후자는 응답이 몇 개 없을 때
        # 모든 키워드가 걸러져 답을 못 만드는 상황을 막는다.
        threshold = min(self.alpha * n, float(self.beta))
        kept = sorted(w for w, c in counter.items() if c >= threshold)

        if not kept:
            # 살아남은 키워드가 없다 = 응답들이 전혀 합의하지 못했다.
            # 지어내는 것보다 모른다고 답하는 편이 안전하다.
            return {"response": "I don't know.", "isolated": list(isolated),
                    "n_responded": n, "threshold": threshold,
                    "keywords": [], "keyword_counts": dict(counter)}

        # 5단계: 키워드만으로 최종 답변. 문서를 다시 넣지 않는 것이 핵심이다 —
        # 여기서 문서를 넣으면 오염이 되살아난다.
        raw = await self._ask(
            _FINAL_PROMPT.format(keywords=", ".join(kept), query=query),
            as_json=True)
        try:
            final = str(json.loads(raw).get("answer") or "").strip()
        except json.JSONDecodeError:
            final = raw          # 스키마가 깨지면 원문이라도 넘긴다
        return {"response": final, "isolated": list(isolated),
                "n_responded": n, "threshold": threshold,
                "keywords": kept, "keyword_counts": dict(counter)}
