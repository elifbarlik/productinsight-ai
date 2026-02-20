from src.rag_core import ProductRAG

print("RAG SYSTEM TEST")
print("-" * 50)

rag = ProductRAG()

test_queries = [
    "Do you have black jackets?",
    "I'm looking for shoes under $100",
    "What's a good formal outfit?",
]

for query in test_queries:
    print(f"\n👤 Query: '{query}'")
    print("-" * 50)

    response, products = rag.query(query)

    print(f"\n🤖 Response:\n{response}")

    print(f"\n📦 Retrieved Products:")
    for p in products:
        print(f"  - {p.name} (${p.price}) - {p.color}")

    print()

print("-" * 50)
print("✓ RAG TEST COMPLETE")