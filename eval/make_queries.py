"""Generate eval/queries.jsonl: queries + ground-truth relevant product ids.

Ground truth is derived *from the catalogue itself* with criteria written here,
independent of the parser/retriever code:

  attribute  - relevance defined by structured columns (colour/category/price)
  feature    - relevance defined by a keyword in the product name/description
  combined   - attribute AND feature
  semantic   - weak labels for intent-style queries (heuristic, see criteria)
  negative   - nothing in the catalogue matches; a good system returns nothing

Re-run after changing the catalogue:  python -m eval.make_queries
"""
import json
from pathlib import Path

import pandas as pd

CSV = Path("src/data/product.csv")
OUT = Path("eval/queries.jsonl")

KURTA_FAMILY = {"Kurta", "Kurta Set"}
BLUE = {"blue", "navy", "teal"}
GREEN = {"green"}
YELLOW = {"yellow", "mustard"}
HEAVY_WORK = ("embroider", "sequin", "gotta patti", "mirror", "thread work")


def toks(color: str) -> set:
    return set(color.lower().replace("-", " ").split())


def text(r) -> str:
    return f"{r['name']} {r['description']}".lower()


def cat_in(r, cats):
    return r["category"] in cats


QUERIES = [
    # ── attribute ──
    ("attribute", "black kurta for women", lambda r: "black" in toks(r["color"]) and cat_in(r, KURTA_FAMILY)),
    ("attribute", "navy blue kurta set", lambda r: r["color"] == "Navy Blue" and r["category"] == "Kurta Set"),
    ("attribute", "teal kurta", lambda r: "teal" in toks(r["color"]) and cat_in(r, KURTA_FAMILY)),
    ("attribute", "pink kurta under 2500", lambda r: "pink" in toks(r["color"]) and cat_in(r, KURTA_FAMILY) and r["price"] <= 2500),
    ("attribute", "red kurta below 2000", lambda r: "red" in toks(r["color"]) and cat_in(r, KURTA_FAMILY) and r["price"] <= 2000),
    ("attribute", "green kurta", lambda r: bool(GREEN & toks(r["color"])) and cat_in(r, KURTA_FAMILY)),
    ("attribute", "sharara set", lambda r: r["category"] == "Sharara Set"),
    ("attribute", "skirt set for women", lambda r: r["category"] == "Skirt Set"),
    ("attribute", "maroon kurta set under 4500", lambda r: bool({"maroon", "burgundy"} & toks(r["color"])) and r["category"] == "Kurta Set" and r["price"] <= 4500),
    ("attribute", "cheap kurta under 1500", lambda r: cat_in(r, KURTA_FAMILY) and r["price"] <= 1500),
    ("attribute", "yellow kurta", lambda r: bool(YELLOW & toks(r["color"])) and cat_in(r, KURTA_FAMILY)),
    ("attribute", "kurta between 3000 and 4000", lambda r: cat_in(r, KURTA_FAMILY) and 3000 <= r["price"] <= 4000),
    ("attribute", "kurta over 5000", lambda r: cat_in(r, KURTA_FAMILY) and r["price"] >= 5000),
    ("attribute", "salwar set", lambda r: r["category"] == "Salwar Set"),
    # ── feature (keyword in name/description) ──
    ("feature", "anarkali kurta", lambda r: "anarkali" in text(r)),
    ("feature", "bandhani print", lambda r: "bandhani" in text(r)),
    ("feature", "gotta patti work", lambda r: "gotta patti" in text(r)),
    ("feature", "mirror work kurta", lambda r: "mirror" in text(r)),
    ("feature", "pure cotton kurta", lambda r: "pure cotton" in text(r)),
    ("feature", "sleeveless kurta", lambda r: "sleeveless" in text(r)),
    ("feature", "mandarin collar", lambda r: "mandarin" in text(r)),
    ("feature", "keyhole neck kurta", lambda r: "keyhole" in text(r)),
    ("feature", "thread work embroidered kurta", lambda r: "thread work" in text(r)),
    ("feature", "floral printed kurta with palazzos", lambda r: "floral" in text(r) and "palazzo" in text(r)),
    ("feature", "ethnic motifs", lambda r: "ethnic motifs" in text(r)),
    ("feature", "striped kurta", lambda r: "striped" in text(r)),
    ("feature", "geometric print", lambda r: "geometric" in text(r)),
    # ── combined (attribute + feature) ──
    ("combined", "red anarkali under 6000", lambda r: "anarkali" in text(r) and "red" in toks(r["color"]) and r["price"] <= 6000),
    ("combined", "pink floral kurta below 3000", lambda r: "pink" in toks(r["color"]) and "floral" in text(r) and cat_in(r, KURTA_FAMILY) and r["price"] <= 3000),
    ("combined", "blue kurta with dupatta under 5000", lambda r: bool(BLUE & toks(r["color"])) and cat_in(r, KURTA_FAMILY) and "dupatta" in text(r) and r["price"] <= 5000),
    ("combined", "teal embroidered kurta", lambda r: "teal" in toks(r["color"]) and "embroider" in text(r) and cat_in(r, KURTA_FAMILY)),
    ("combined", "black printed kurta below 2000", lambda r: "black" in toks(r["color"]) and "printed" in text(r) and cat_in(r, KURTA_FAMILY) and r["price"] <= 2000),
    ("combined", "navy blue cotton kurta set", lambda r: r["color"] == "Navy Blue" and "cotton" in text(r) and r["category"] == "Kurta Set"),
    # ── semantic (weak labels: heuristic intent) ──
    ("semantic", "elegant heavily embellished outfit for a wedding", lambda r: any(w in text(r) for w in HEAVY_WORK) and r["price"] >= 3500),
    ("semantic", "light breathable outfit for summer", lambda r: "cotton" in text(r)),
    ("semantic", "festive party wear with dupatta", lambda r: "dupatta" in text(r) and any(w in text(r) for w in HEAVY_WORK)),
    ("semantic", "affordable casual daily wear kurta", lambda r: cat_in(r, KURTA_FAMILY) and r["price"] <= 2000),
    # ── negative (catalogue has nothing relevant) ──
    ("negative", "black leather jacket", lambda r: False),
    ("negative", "men's sneakers", lambda r: False),
    ("negative", "formal suit for men", lambda r: False),
    ("negative", "blue denim jeans", lambda r: False),
    ("negative", "gold necklace", lambda r: False),
    ("negative", "kurta under 500", lambda r: False),
    ("negative", "kurta for men", lambda r: False),
]


def main():
    df = pd.read_csv(CSV)
    rows = df.to_dict("records")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for qtype, query, truth in QUERIES:
            relevant = [int(r["product_id"]) for r in rows if truth(r)]
            if qtype != "negative" and not relevant:
                raise SystemExit(f"query has no relevant products, fix the criteria: {query!r}")
            if qtype == "negative" and relevant:
                raise SystemExit(f"negative query has relevant products: {query!r}")
            f.write(json.dumps({"query": query, "type": qtype, "relevant": relevant}) + "\n")
            print(f"{qtype:9s} {len(relevant):3d}  {query}")
    print(f"\nwrote {len(QUERIES)} queries -> {OUT}")


if __name__ == "__main__":
    main()
