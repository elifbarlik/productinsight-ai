"""Synthetic FAISS benchmark: IndexFlatIP (exact) vs IndexHNSWFlat (approximate).

    python -m eval.benchmark_index --n 50000 --dim 768

Uses clustered random unit vectors (a rough stand-in for real embeddings) because the
real catalogue (~50 items) is far too small to show any difference. Numbers therefore
show *scaling behaviour*, not retrieval quality on product data.
"""
import argparse
import time

import faiss
import numpy as np


def clustered_vectors(n, dim, clusters, rng):
    centers = rng.normal(size=(clusters, dim)).astype("float32")
    assign = rng.integers(0, clusters, size=n)
    x = centers[assign] + 0.6 * rng.normal(size=(n, dim)).astype("float32")
    faiss.normalize_L2(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    data = clustered_vectors(args.n, args.dim, 200, rng)
    queries = clustered_vectors(args.queries, args.dim, 200, rng)

    flat = faiss.IndexFlatIP(args.dim)
    flat.add(data)
    t = time.perf_counter()
    for q in queries:
        _, truth = flat.search(q[None], args.k)
    flat_ms = (time.perf_counter() - t) * 1000 / args.queries
    _, exact = flat.search(queries, args.k)

    print(f"n={args.n} dim={args.dim} k={args.k} queries={args.queries}\n")
    print("| index | build s | ms/query | recall@k vs exact |\n|---|---|---|---|")
    print(f"| flat (exact) | - | {flat_ms:.2f} | 1.000 |")

    for ef in (16, 64, 128, 256, 512):
        hnsw = faiss.IndexHNSWFlat(args.dim, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = 100
        hnsw.hnsw.efSearch = ef
        t = time.perf_counter()
        hnsw.add(data)
        build_s = time.perf_counter() - t
        t = time.perf_counter()
        for q in queries:
            hnsw.search(q[None], args.k)
        ms = (time.perf_counter() - t) * 1000 / args.queries
        _, approx = hnsw.search(queries, args.k)
        recall = np.mean([len(set(a) & set(e)) / args.k for a, e in zip(approx, exact)])
        print(f"| hnsw (M=32, efSearch={ef}) | {build_s:.1f} | {ms:.2f} | {recall:.3f} |")


if __name__ == "__main__":
    main()
