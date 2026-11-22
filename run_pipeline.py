"""Run the same PDF, Chroma and answer path used by the app.

Examples:
  python run_pipeline.py data/papers/chexpert.pdf --question "How many patients?"
  python run_pipeline.py data/papers/chexpert.pdf --questions evaluations/chexpert_questions.json
"""
import argparse
import json
from pathlib import Path

from rag.native import create_vectorstore, load_and_chunk
from rag.local_llm import LocalOllama
from rag.structured import ask_structured
from evaluate import evaluate_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question")
    group.add_argument("--questions", type=Path)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    chunks = load_and_chunk(str(args.pdf))
    store = create_vectorstore(chunks, persist_directory=None)
    cases = (json.loads(args.questions.read_text(encoding="utf-8")) if args.questions
             else [{"question": args.question}])
    retrieval = evaluate_cases(store, cases) if args.questions else None
    model = LocalOllama(args.model)
    answers = []
    for case in cases:
        try:
            answer, sources = ask_structured(store, case["question"], model)
            answers.append({"question": case["question"], "answer": answer.answer,
                            "found": answer.found, "source_pages": answer.source_pages,
                            "evidence_quotes": answer.evidence_quotes,
                            "retrieved_pages": [doc.metadata.get("page") for doc in sources]})
        except (ValueError, OSError, KeyError) as exc:
            answers.append({"question": case["question"], "error": str(exc)})
    report = {"pdf": str(args.pdf), "model": args.model, "chunks": len(chunks),
              "retrieval": retrieval, "answers": answers}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

