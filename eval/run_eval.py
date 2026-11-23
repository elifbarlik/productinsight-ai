"""Retrieval evaluation harness.

    python -m eval.run_eval                          # all configurations that can run
    python -m eval.run_eval --configs bm25,bm25+filter
    python -m eval.run_eval --sweep-threshold        # tune MIN_SIMILARITY (needs API key + index)
    python -m eval.run_eval --judge                  # + LLM-as-judge on generated answers

A configuration is "<mode>" or "<mode>+filter" where mode is bm25 | dense | hybrid and
"+filter" enables structured query parsing + hard metadata filtering.
Dense / hybrid configurations need GEMINI_API_KEY and a built index; they are skipped
(and reported as such) when unavailable.
"""
import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from eval.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from src.catalog import ProductCatalog
from src.config import (
    faiss_image_path,
    faiss_table_path,
    faiss_text_path,
    gemini_api_key,
    metadata_path,
    product_csv_path,
)
from src.query_parser import RuleBasedParser
from src.retriever import Retriever

QUERIES = Path("eval/queries.jsonl")
RESULTS = Path("eval/results")
ALL_CONFIGS = ["bm25", "bm25+filter", "dense", "dense+filter", "hybrid", "hybrid+filter"]
TYPES = ["attribute", "feature", "combined", "semantic"]


def load_queries() -> List[dict]:
    return [json.loads(line) for line in QUERIES.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_retriever_for_eval(min_sim: float = None):
    """Returns (retriever, dense_ok). Falls back to BM25-only when no index/key."""
    kwargs = {} if min_sim is None else {"min_sim": min_sim}
    if gemini_api_key and os.path.exists(metadata_path):
        from src.embedding_utils import GeminiEmbedder
        from src.vector_store import IndexFormatError, MultimodalVectorStore

        try:
            store = MultimodalVectorStore.load(
                faiss_text_path, faiss_image_path, faiss_table_path, metadata_path
            )
            return Retriever(store.metadata, store, GeminiEmbedder(), RuleBasedParser(), **kwargs), True
        except IndexFormatError as exc:
            print(f"[skip dense] {exc}")
    catalog = ProductCatalog(product_csv_path)
    return Retriever(catalog.products, parser=RuleBasedParser(), **kwargs), False


def evaluate(retriever: Retriever, queries: List[dict], config: str, k: int = 10) -> dict:
    mode, _, flag = config.partition("+")
    use_filter = flag == "filter"
    per_query = []
    for q in queries:
        result = retriever.search(q["query"], k=k, mode=mode, use_filter=use_filter)
        ranked = [p.product_id for p in result.products]
        rel = set(q["relevant"])
        row = {"query": q["query"], "type": q["type"], "ranked": ranked, "n_relevant": len(rel)}
        if q["type"] == "negative":
            row["rejected"] = len(ranked) == 0
        else:
            row.update(
                p5=precision_at_k(ranked, rel, 5),
                r10=recall_at_k(ranked, rel, 10),
                mrr=reciprocal_rank(ranked, rel),
                ndcg10=ndcg_at_k(ranked, rel, 10),
            )
        row["latency_ms"] = round(sum(result.timings.values()), 2)
        per_query.append(row)
    return {"config": config, "per_query": per_query, "summary": summarize(per_query)}


def _mean(rows, key):
    vals = [r[key] for r in rows]
    return round(statistics.mean(vals), 3) if vals else None


def summarize(per_query: List[dict]) -> dict:
    positives = [r for r in per_query if r["type"] != "negative"]
    negatives = [r for r in per_query if r["type"] == "negative"]
    out = {
        "overall": {m: _mean(positives, m) for m in ("p5", "r10", "mrr", "ndcg10")},
        "neg_reject_rate": _mean([{"v": float(r["rejected"])} for r in negatives], "v"),
        "latency_ms_mean": _mean(per_query, "latency_ms"),
        "by_type": {},
    }
    by_type = defaultdict(list)
    for r in positives:
        by_type[r["type"]].append(r)
    for t in TYPES:
        if t in by_type:
            out["by_type"][t] = {m: _mean(by_type[t], m) for m in ("p5", "r10", "mrr", "ndcg10")}
    return out


def to_markdown(results: List[dict], skipped: List[str], n_queries: int) -> str:
    lines = [
        f"Evaluation over {n_queries} queries (`eval/queries.jsonl`); "
        "P@5 / Recall@10 / MRR / nDCG@10 averaged over positive queries; "
        "`neg-reject` = share of out-of-catalogue queries answered with no results.",
        "",
        "| config | P@5 | R@10 | MRR | nDCG@10 | neg-reject | ms/query |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s, o = r["summary"], r["summary"]["overall"]
        lines.append(
            f"| {r['config']} | {o['p5']} | {o['r10']} | {o['mrr']} | {o['ndcg10']} "
            f"| {s['neg_reject_rate']} | {s['latency_ms_mean']} |"
        )
    lines += ["", "Per query type (nDCG@10 / MRR):", "", "| config | " + " | ".join(TYPES) + " |",
              "|---|" + "---|" * len(TYPES)]
    for r in results:
        cells = [
            f"{r['summary']['by_type'][t]['ndcg10']} / {r['summary']['by_type'][t]['mrr']}"
            if t in r["summary"]["by_type"] else "-"
            for t in TYPES
        ]
        lines.append(f"| {r['config']} | " + " | ".join(cells) + " |")
    if skipped:
        lines += ["", "Not run (needs `GEMINI_API_KEY` + built index): " + ", ".join(skipped)]
    return "\n".join(lines)


def sweep_threshold(queries: List[dict], k: int):
    retriever, dense_ok = build_retriever_for_eval()
    if not dense_ok:
        raise SystemExit("--sweep-threshold needs GEMINI_API_KEY and a built index")
    print("| MIN_SIMILARITY | nDCG@10 | MRR | neg-reject |\n|---|---|---|---|")
    for t in (0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        retriever.min_sim = t
        s = evaluate(retriever, queries, "hybrid+filter", k)["summary"]
        print(f"| {t} | {s['overall']['ndcg10']} | {s['overall']['mrr']} | {s['neg_reject_rate']} |")


def judge_answers(retriever: Retriever, queries: List[dict]):
    """LLM-as-judge: grounded rate + helpfulness (1-5) of generated answers."""
    from pydantic import BaseModel

    from src.llm import GeminiLLM
    from src.rag_core import ProductRAG, format_context

    class Verdict(BaseModel):
        helpfulness: int  # 1-5
        accurate: bool    # every claim is supported by the context

    llm = GeminiLLM()
    rag = ProductRAG(retriever, llm)
    scores, grounded, accurate, llm_used, failed = [], [], [], [], 0
    for q in queries:
        try:
            resp = rag.answer(q["query"])
            ctx = format_context(resp.search.products) if resp.search.products else "(no products retrieved)"
            verdict = Verdict.model_validate(llm.generate_json(
                "You grade a shopping assistant. Score helpfulness 1-5 for the customer question and say "
                "whether every claim in the answer is supported by the context.",
                f"Context:\n{ctx}\n\nQuestion: {q['query']}\n\nAnswer: {resp.message}", Verdict))
        except Exception as exc:  # noqa: BLE001 - one failed call must not discard the whole run
            failed += 1
            print(f"[judge] skipped {q['query']!r}: {str(exc)[:80]}")
            continue
        grounded.append(resp.grounded)
        llm_used.append(resp.llm_used)
        scores.append(verdict.helpfulness)
        accurate.append(verdict.accurate)
    n = len(scores)
    if not n:
        raise SystemExit("judge produced no results (LLM unavailable?)")
    print(f"LLM judge ({llm.model}) over {n}/{len(queries)} queries ({failed} failed): "
          f"helpfulness={statistics.mean(scores):.2f}/5, judge-accurate={sum(accurate) / n:.0%}, "
          f"id-grounded={sum(grounded) / n:.0%}, answers generated by LLM (not fallback)={sum(llm_used) / n:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(ALL_CONFIGS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sweep-threshold", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--out", default="latest")
    args = ap.parse_args()

    queries = load_queries()
    if args.sweep_threshold:
        return sweep_threshold(queries, args.k)

    retriever, dense_ok = build_retriever_for_eval()
    if args.judge:
        if not dense_ok:
            raise SystemExit("--judge needs GEMINI_API_KEY and a built index")
        return judge_answers(retriever, queries)

    results, skipped = [], []
    for config in args.configs.split(","):
        if not config.startswith("bm25") and not dense_ok:
            skipped.append(config)
            continue
        results.append(evaluate(retriever, queries, config, args.k))

    md = to_markdown(results, skipped, len(queries))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.out}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (RESULTS / f"{args.out}.md").write_text(md + "\n", encoding="utf-8")
    print(md)

    worst: Dict[str, List[str]] = {}
    for r in results:
        worst[r["config"]] = [
            q["query"] for q in r["per_query"] if q["type"] != "negative" and q["ndcg10"] == 0
        ]
    for config, qs in worst.items():
        if qs:
            print(f"\n[{config}] zero-score queries: {qs}")


if __name__ == "__main__":
    main()
