"""Evaluate retrieval on a manually checked question set.

Usage: python evaluate.py data/papers/chexpert.pdf evaluations/questions.json
Gold entries require a question, a zero-based source page, and an exact answer
evidence substring copied from that page. This measures retrieval, not whether
Claude produces a factually correct answer.
"""
import argparse
import json
from pathlib import Path

def evaluate_cases(vectorstore, cases, k=3):
    results = []
    for case in cases:
        if not case.get("question") or not case.get("evidence") or not isinstance(case.get("page"), int):
            raise ValueError("Each case needs question, evidence, and integer zero-based page")
        hits = vectorstore.similarity_search(case["question"], k=k)
        ranks = [i + 1 for i, doc in enumerate(hits)
                 if case["evidence"].casefold() in doc.page_content.casefold()]
        rank = ranks[0] if ranks else None
        results.append({"question": case["question"], "evidence_rank": rank,
                        "retrieved_pages": [doc.metadata.get("page") for doc in hits]})
    n = len(results)
    if not n:
        raise ValueError("Evaluation set must contain at least one case")
    return {"cases": n, "recall_at_1": sum(x["evidence_rank"] == 1 for x in results) / n,
            "recall_at_3": sum(x["evidence_rank"] is not None and x["evidence_rank"] <= 3 for x in results) / n,
            "mrr_at_3": sum(1 / x["evidence_rank"] for x in results if x["evidence_rank"]) / n,
            "details": results}


def main():
    from rag.loader import load_and_chunk
    from rag.vectorstore import get_embeddings
    from langchain_chroma import Chroma
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluations/retrieval_results.json"))
    args = parser.parse_args()
    cases = json.loads(args.questions.read_text(encoding="utf-8"))
    chunks = load_and_chunk(str(args.pdf))
    for case in cases:
        if not any(chunk.metadata.get("page") == case.get("page")
                   and case.get("evidence", "").casefold() in chunk.page_content.casefold()
                   for chunk in chunks):
            raise ValueError(f"Gold evidence not present in a single indexed chunk: {case.get('question')}")
    # Fresh in-memory index prevents earlier uploads from leaking into this run.
    vectorstore = Chroma.from_documents(documents=chunks, embedding=get_embeddings())
    report = evaluate_cases(vectorstore, cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: val for key, val in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()

