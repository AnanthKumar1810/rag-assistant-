"""Generate paired local-model answers for manual grounding review.

This is a fixed-window MiniLM/Ollama experiment, separate from the Claude app.
"""
import json
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

root = Path(__file__).resolve().parents[1]
cases = json.loads((root / "evaluations/chexpert_questions.json").read_text(encoding="utf-8"))[:3]
pages = [p.extract_text() or "" for p in PdfReader(root / "data/papers/chexpert.pdf").pages]
chunks = [(i, text[start:start+1000]) for i,text in enumerate(pages)
          for start in range(0,len(text),800) if text[start:start+1000]]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",local_files_only=True)
vectors = model.encode([t for _,t in chunks],normalize_embeddings=True)
queries = model.encode([c["question"] for c in cases],normalize_embeddings=True)
out=[]
for case,query in zip(cases,queries):
    ranked=np.argsort(-(vectors @ query))
    for k in (1,3,4):
        selected=[chunks[idx] for idx in ranked[:k]]
        context="\n\n".join(f"[page {page}] {content}" for page,content in selected)
        prompt=("Answer only from the passages. If the answer is absent, say 'Not found in passages.' "
                "Do not guess.\n"+context+"\nQuestion: "+case["question"])
        payload={"model":"qwen2.5:3b","stream":False,"options":{"temperature":0},
                 "messages":[{"role":"user","content":prompt}]}
        req=Request("http://127.0.0.1:11434/api/chat",json.dumps(payload).encode(),
                    {"Content-Type":"application/json"})
        with build_opener(ProxyHandler({})).open(req,timeout=120) as response:
            answer=json.load(response)["message"]["content"]
        out.append({"question":case["question"],"gold_evidence":case["evidence"],"k":k,
                    "evidence_in_context":any(case["evidence"] in t for _,t in selected),
                    "retrieved_pages":[p for p,_ in selected],"answer":answer})
        print(case["question"],k,answer,flush=True)
(root/"evaluations/paired_answer_result.json").write_text(json.dumps(out,indent=2),encoding="utf-8")

