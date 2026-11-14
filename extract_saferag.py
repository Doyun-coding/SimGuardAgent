"""
SafeRAG NCTD 데이터셋에서 오염된 샘플 추출
nctd.json → JSON 파일로 저장

Attack types:
  SN   (Silver Noise)         — 100개
  WDoS (White DoS)            — 100개
  SA   (Soft Ad)              — 94개
  ICC  (Inter-Context Conflict) — 93개

사용법:
  python extract_saferag.py [--limit N] [--out PATH]
  python extract_saferag.py --limit 100   # 타입별 최대 100개
  python extract_saferag.py --out results/saferag.json
"""
import argparse
import json
import os

NCTD_PATH = os.path.join(os.path.dirname(__file__), "SafeRAG", "nctd_datasets", "nctd.json")

ATTACK_TYPES = ["SN", "WDoS", "SA", "ICC"]

ATTACK_LABELS = {
    "SN":   "Silver Noise",
    "WDoS": "White DoS",
    "SA":   "Soft Ad",
    "ICC":  "Inter-Context Conflict",
}


def load_nctd(path: str = NCTD_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_prompt(item: dict, attack_type: str) -> str:
    """질문 + 오염된 컨텍스트를 합쳐 프롬프트 생성"""
    ctx_key = f"enhanced_{attack_type}_contexts"
    contexts = item.get(ctx_key, item.get("enhanced_contexts", []))
    ctx_str = "\n\n".join(str(c) for c in contexts[:3])  # 상위 3개 컨텍스트
    return (
        f"다음 문서를 참고하여 질문에 답하세요.\n\n"
        f"[검색된 문서]\n{ctx_str}\n\n"
        f"[질문]\n{item['questions']}"
    )


def extract(limit: int = 100) -> list[dict]:
    data = load_nctd()
    results = []
    for attack_type in ATTACK_TYPES:
        items = data.get(attack_type, [])[:limit]
        for item in items:
            results.append({
                "attack_type":  attack_type,
                "attack_label": ATTACK_LABELS[attack_type],
                "id":           item.get("id", ""),
                "question":     item["questions"],
                "prompt":       build_prompt(item, attack_type),
                "attack_kws":   item.get("attack_kws", []),
                "poisoned_contexts": item.get(f"enhanced_{attack_type}_contexts", []),
                "clean_contexts":    item.get("enhanced_contexts", []),
                "label": 1,  # 모두 attack 샘플
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="타입별 최대 샘플 수 (default: 100)")
    parser.add_argument("--out", default="results/saferag/nctd_poisoned.json",
                        help="출력 경로 (default: results/saferag/nctd_poisoned.json)")
    args = parser.parse_args()

    samples = extract(args.limit)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # 통계 출력
    from collections import Counter
    counts = Counter(s["attack_type"] for s in samples)
    print(f"\n=== SafeRAG NCTD 추출 완료 ===")
    print(f"총 {len(samples)}개 저장: {args.out}")
    for at in ATTACK_TYPES:
        print(f"  {at:5s} ({ATTACK_LABELS[at]:25s}): {counts.get(at, 0)}개")


if __name__ == "__main__":
    main()
