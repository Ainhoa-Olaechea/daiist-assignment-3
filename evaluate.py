"""
Assignment 3: evaluation (`main.py evaluate`).

Stage 1 - retrieval: hit rate at k = 1, 3, 5 for each chunking design (article vs fixed).
  A question is a hit at k if one of its expected (source, location) pairs is among the
  top-k chunks. For multi questions we also report "all@k": every expected pair retrieved.
  Unanswerable questions have no expected passage and are left out of this stage.
Stage 2 - end-to-end (agent answers, code checks, LLM judge): added in a later step.

Outputs (committed): eval/retrieval_results.csv, eval/retrieval_summary.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import index as ix

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "eval"
KS = (1, 3, 5)


def load_questions() -> list[dict]:
    with open(EVAL_DIR / "questions.csv", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def expected_pairs(q: dict) -> set[tuple[str, str]]:
    sources = [s.strip() for s in q["expected_source"].split(";")]
    locations = [l.strip() for l in q["expected_location"].split(";")]
    return set(zip(sources, locations))


def hit_pairs(hit: dict) -> set[tuple[str, str]]:
    return {(hit["doc_id"], loc) for loc in hit["locations"].split("|")}


def first_rank(hits: list[dict], expected: set) -> int | None:
    """Rank of the first chunk that contains any expected passage."""
    for rank, hit in enumerate(hits, start=1):
        if hit_pairs(hit) & expected:
            return rank
    return None


def all_rank(hits: list[dict], expected: set) -> int | None:
    """Smallest k at which every expected passage has been retrieved."""
    found = set()
    for rank, hit in enumerate(hits, start=1):
        found |= hit_pairs(hit) & expected
        if found == expected:
            return rank
    return None


def rate(ranks: list[int | None], k: int) -> float:
    return round(sum(1 for r in ranks if r is not None and r <= k) / len(ranks), 3) if ranks else 0.0


def retrieval_eval() -> dict:
    questions = [q for q in load_questions() if q["kind"] != "unanswerable"]
    rows, summary = [], {}
    for design in ix.DESIGNS:
        if ix.collection(design).count() == 0:
            print(f"[{design}] collection is empty; skipped")
            continue
        any_ranks, multi_all = [], []
        by_kind: dict[str, list] = {}
        for q in questions:
            expected = expected_pairs(q)
            hits = ix.search(q["question"], k=max(KS), design=design)
            r_any, r_all = first_rank(hits, expected), all_rank(hits, expected)
            any_ranks.append(r_any)
            by_kind.setdefault(q["kind"], []).append(r_any)
            if q["kind"] == "multi":
                multi_all.append(r_all)
            rows.append({"design": design, "id": q["id"], "kind": q["kind"],
                         "expected": q["expected_source"] + " | " + q["expected_location"],
                         "first_hit_rank": r_any or "", "all_found_rank": r_all or "",
                         "top5": " ; ".join(h["citation"] for h in hits)})
        summary[design] = {
            "questions": len(any_ranks),
            **{f"hit@{k}": rate(any_ranks, k) for k in KS},
            **{f"{kind}_hit@{k}": rate(r, k) for kind, r in by_kind.items() for k in KS},
            **{f"multi_all@{k}": rate(multi_all, k) for k in KS},
        }

    with open(EVAL_DIR / "retrieval_results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["design"])
        writer.writeheader()
        writer.writerows(rows)
    (EVAL_DIR / "retrieval_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nRetrieval hit rate (answerable questions)")
    print(f"{'design':<10}" + "".join(f"{'hit@' + str(k):>9}" for k in KS) + "".join(f"{'multi all@' + str(k):>14}" for k in KS))
    for design, s in summary.items():
        print(f"{design:<10}" + "".join(f"{s[f'hit@{k}']:>9.0%}" for k in KS)
              + "".join(f"{s[f'multi_all@{k}']:>14.0%}" for k in KS))
    if summary:
        best = max(summary, key=lambda d: (summary[d]["hit@3"], summary[d]["hit@1"], summary[d]["hit@5"]))
        print(f"\nBest by hit@3 (ties: hit@1, hit@5): {best}  |  design used by the server now: {ix.active_design()}")
    return summary


def main() -> int:
    try:
        retrieval_eval()
    except ix.DailyQuotaReached:
        print("Daily embedding limit reached while embedding questions. "
              "Query vectors so far are cached; run again after 09:00 (Spain).")
    return 0


if __name__ == "__main__":
    sys.exit(main())