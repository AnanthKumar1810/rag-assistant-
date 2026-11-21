"""Compare fixed-character chunk settings with a frozen source-evidence set."""
import json
from pathlib import Path
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

root = Path(__file__).resolve().parents[1]
cases = json.loads((root / "evaluations/chexpert_questions.json").read_text(encoding="utf-8"))
pages = [p.extract_text() or "" for p in PdfReader(root / "data/papers/chexpert.pdf").pages]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
queries = model.encode([c["question"] for c in cases], normalize_embeddings=True)
out = []
for size, overlap in [(500, 100), (1000, 200), (1500, 300)]:
    chunks = [(i, text[start:start+size]) for i, text in enumerate(pages)
              for start in range(0, len(text), size-overlap) if text[start:start+size]]
    vectors = model.encode([t for _, t in chunks], normalize_embeddings=True)
    ranks = []
    for case, query in zip(cases, queries):
        ranking = np.argsort(-(vectors @ query))
        ranks.append(next((n for n, idx in enumerate(ranking, 1)
                           if chunks[idx][0] == case["page"] and case["evidence"] in chunks[idx][1]), None))
    out.append({"chunk_size":size,"overlap":overlap,"chunks":len(chunks),
                "recall_at_1":sum(r == 1 for r in ranks)/len(ranks),
                "recall_at_3":sum(r is not None and r <= 3 for r in ranks)/len(ranks),
                "ranks":ranks})
(root / "evaluations/chunk_sweep_result.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))

