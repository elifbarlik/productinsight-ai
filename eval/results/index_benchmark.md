n=50000 dim=768 k=10 queries=200

| index | build s | ms/query | recall@k vs exact |
|---|---|---|---|
| flat (exact) | - | 4.16 | 1.000 |
| hnsw (M=32, efSearch=16) | 4.5 | 0.17 | 0.189 |
| hnsw (M=32, efSearch=64) | 4.4 | 0.38 | 0.443 |
| hnsw (M=32, efSearch=128) | 4.4 | 0.63 | 0.642 |
| hnsw (M=32, efSearch=256) | 4.4 | 1.12 | 0.828 |
| hnsw (M=32, efSearch=512) | 4.4 | 1.88 | 0.955 |
