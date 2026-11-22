"""Offline MiniLM retrieval probe without the app's LangChain/Chroma dependencies.

Uses fixed 1000-character windows with 200-character overlap, unlike the app's
recursive splitter. Keep this result separate from the production RAG index.
"""
import json
from pathlib import Path
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

root = Path(__file__).resolve().parents[1]
cases = json.loads((root / "evaluations/chexpert_questions.json").read_text(encoding="utf-8"))
pages = [p.extract_text() or "" for p in PdfReader(root / "data/papers/chexpert.pdf").pages]
chunks = [(i, text[start:start + 1000]) for i, text in enumerate(pages)
          for start in range(0, len(text), 800) if text[start:start + 1000]]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
vectors = model.encode([text for _, text in chunks], normalize_embeddings=True)
queries = model.encode([c["question"] for c in cases], normalize_embeddings=True)
details = []
for case, query in zip(cases, queries):
    full_ranking = np.argsort(-(vectors @ query))
    ranking = full_ranking[:3]
    rank = next((n for n, idx in enumerate(full_ranking, 1)
                 if case["evidence"] in chunks[idx][1]), None)
    details.append({"question": case["question"], "evidence_rank": rank,
                    "pages": [chunks[idx][0] for idx in ranking]})
n = len(details)
result = {"protocol": "fixed-character-window offline probe; not LangChain recursive splitting or Chroma",
          "cases": n, "chunks": len(chunks),
          "recall_at_1": sum(d["evidence_rank"] == 1 for d in details) / n,
          "recall_at_3": sum(d["evidence_rank"] is not None and d["evidence_rank"] <= 3 for d in details) / n,
          "mrr_at_3": sum(1 / d["evidence_rank"] for d in details
                          if d["evidence_rank"] is not None and d["evidence_rank"] <= 3) / n,
          "details": details}
(root / "evaluations/offline_probe_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))

