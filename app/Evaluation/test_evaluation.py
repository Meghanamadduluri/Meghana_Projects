from app.db.vector_store import VectorStore

vs = VectorStore()

test_cases = [
    {
        "query": "What are the causes of unfairness?",
        "expected_keywords": "[bias, discrimination, lack of diversity, data quality issues]"
    },
    {
        "query": "What is fairness gerrymandering?",
        "epected_keywords" : ["statistical definitions of fairness", "malicious intent"]
    }, 
    {
        "query": "What is the limitation of crowdsourcing framework?",
        "expected_keywords": ["crowdsourcing framework", "no guidance", "find improvements"]
    }, 
    {
        "query": "What are competition platforms?",
        "expected": "popular framework for generating accurate machine learning models through communal efforts"
    }
]

top1_hits = 0
topk_hits = 0
total = len(test_cases)

for case in test_cases:

    results = vs.hybrid_search(case["query"], n_results=3)

    top_1_text = results[0]["text"].lower()
    top_k_text = " ".join([r["text"] for r in results]).lower()

    keywords = case["expected_keywords"]

    # Top-1 check
    if sum(keyword in top_1_text for keyword in keywords) >= 2:
        top1_hits += 1

    # Top-K check
    if sum(keyword in top_k_text for keyword in keywords) >= 2:
        topk_hits += 1
        print(f"PASS: {case['query']}")
    else:
        print(f"FAIL: {case['query']}")

    print("\nRetrieved Sources:")
    for r in results:
        print("SOURCE:", r["text"])

print("\n=== Evaluation Summary ===")
print(f"Top-1 Accuracy: {top1_hits}/{total} = {top1_hits/total:.2f}")
print(f"Top-K Recall: {topk_hits}/{total} = {topk_hits/total:.2f}")