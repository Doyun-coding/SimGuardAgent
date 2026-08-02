"""응답에서 고유 키워드를 뽑는다 (Algorithm 1의 GetUniqKeywords).

원 논문은 영어 전용이라 spaCy 명사구를 쓴다. SafeRAG는 중국어이므로 언어별로
경로를 나눈다. 중국어는 jieba 형태소 분석, 그 외는 단어 토큰화를 쓴다.

숫자·퍼센트를 반드시 키워드로 남기는 것이 중요하다. ICC 공격이 수치를 조작하는
방식이라('17%→29.7%'를 '10%→27.9%'로), 숫자가 빠지면 조작을 걸러낼 수단이 없다.
"""
from __future__ import annotations

import re

# 한 응답 안에서 같은 단어가 여러 번 나와도 1회로만 센다(원 논문 Line 8).
# 그래야 공격자가 한 문서로 키워드 카운트를 임의로 부풀릴 수 없다.

_CJK = r"一-鿿㐀-䶿"
_HAS_CJK = re.compile(f"[{_CJK}]")

# 숫자·퍼센트·연도. 형태소 분석기가 쪼개기 쉬워 따로 먼저 뽑는다.
_NUMERIC = re.compile(r"\d+(?:\.\d+)?%?")

# 라틴 문자 단어 (중국어 문서에도 영문 고유명사·약어가 섞인다)
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")

# 의미 없는 고빈도어. 이들이 키워드로 남으면 카운트가 포화되어 필터가 무력해진다.
_STOP_ZH = {
    "的", "了", "是", "在", "和", "与", "及", "或", "等", "为", "对", "从", "到",
    "这", "那", "有", "并", "而", "但", "也", "都", "被", "把", "个", "中", "上",
    "下", "他", "她", "它", "我", "你", "们", "会", "能", "可以", "进行", "通过",
    "以及", "根据", "因为", "所以", "如果", "可能", "已经", "没有", "不是", "这些",
    "其中", "同时", "此外", "因此", "以下", "方面", "情况", "问题", "工作", "重要",
}
_STOP_EN = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "not", "no", "we",
    "you", "they", "he", "she", "i", "can", "could", "will", "would", "should",
    "may", "might", "must", "have", "has", "had", "do", "does", "did", "there",
    "here", "what", "which", "who", "when", "where", "how", "why", "all", "any",
    "some", "such", "than", "then", "also", "more", "most", "other", "into",
    "about", "over", "after", "before", "between", "during", "based", "provided",
    "according", "information", "answer", "query", "question", "documents",
    "document", "passage", "passages", "retrieved", "context", "know",
}

_jieba = None


def _get_jieba():
    """jieba 지연 로딩. 사전 빌드가 0.5초쯤 걸려 첫 호출로 미룬다."""
    global _jieba
    if _jieba is None:
        import jieba
        jieba.setLogLevel(60)          # 사전 빌드 로그 억제
        _jieba = jieba
    return _jieba


def extract_keywords(text: str, min_len: int = 2) -> set[str]:
    """응답 하나에서 고유 키워드 집합을 반환한다.

    반환값이 set인 것이 알고리즘상 중요하다 — 중복을 세지 않아야 오염된 문서
    하나가 키워드 카운트를 최대 1만 올릴 수 있다.
    """
    text = (text or "").strip()
    if not text:
        return set()

    words: set[str] = set(_NUMERIC.findall(text))

    if _HAS_CJK.search(text):
        jieba = _get_jieba()
        for w in jieba.cut(text):
            w = w.strip()
            if not w or w in _STOP_ZH:
                continue
            if _HAS_CJK.search(w):
                # 중국어는 1글자 단어 대부분이 조사·기능어라 2글자부터 센다
                if len(w) >= min_len:
                    words.add(w)
            elif w.isalpha() and len(w) >= min_len and w.lower() not in _STOP_EN:
                words.add(w.lower())
    else:
        for w in _LATIN.findall(text):
            wl = w.lower()
            if len(wl) >= min_len and wl not in _STOP_EN:
                words.add(wl)

    return words
