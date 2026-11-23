Evaluation over 44 queries (`eval/queries.jsonl`); P@5 / Recall@10 / MRR / nDCG@10 averaged over positive queries; `neg-reject` = share of out-of-catalogue queries answered with no results.

| config | P@5 | R@10 | MRR | nDCG@10 | neg-reject | ms/query |
|---|---|---|---|---|---|---|
| bm25 | 0.605 | 0.817 | 0.824 | 0.776 | 0.143 | 0.212 |
| bm25+filter | 0.87 | 0.854 | 0.922 | 0.878 | 1.0 | 0.108 |
| dense | 0.573 | 0.776 | 0.83 | 0.773 | 0.571 | 0.369 |
| dense+filter | 0.777 | 0.826 | 0.932 | 0.859 | 1.0 | 0.319 |
| hybrid | 0.568 | 0.835 | 0.883 | 0.831 | 0.143 | 0.497 |
| hybrid+filter | 0.788 | 0.881 | 0.959 | 0.914 | 1.0 | 0.349 |

Per query type (nDCG@10 / MRR):

| config | attribute | feature | combined | semantic |
|---|---|---|---|---|
| bm25 | 0.721 / 0.764 | 0.985 / 1.0 | 0.883 / 0.917 | 0.127 / 0.323 |
| bm25+filter | 1.0 / 1.0 | 0.933 / 1.0 | 1.0 / 1.0 | 0.089 / 0.281 |
| dense | 0.789 / 0.764 | 0.769 / 0.923 | 0.89 / 0.833 | 0.56 / 0.75 |
| dense+filter | 1.0 / 1.0 | 0.763 / 0.923 | 0.938 / 0.917 | 0.56 / 0.75 |
| hybrid | 0.789 / 0.798 | 0.949 / 1.0 | 0.926 / 0.917 | 0.456 / 0.75 |
| hybrid+filter | 1.0 / 1.0 | 0.922 / 1.0 | 1.0 / 1.0 | 0.458 / 0.625 |
