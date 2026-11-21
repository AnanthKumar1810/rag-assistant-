# Retrieval and answer review

Create `questions.json` as an array of independently written questions:

```json
[
  {"question": "What is the paper's primary dataset?", "page": 0,
   "evidence": "an exact short span copied from the supporting page"}
]
```

The `page` value is zero-based, as in PyPDFLoader metadata. Keep the evidence
span within one 1,000-character chunk. Write questions and select supporting
passages from the original PDF before examining retrieval results. Include
questions requiring evidence near chunk boundaries and questions whose answer
is absent. Record absent-answer behavior separately; this script requires a
positive evidence passage for retrieval ranking.

Run `python evaluate.py data/papers/chexpert.pdf evaluations/questions.json`.
The report records evidence rank, recall@1, recall@3 and MRR@3. It measures
whether retrieval found the specified evidence, not generated-answer quality.
For answer quality, record the question, model answer, cited passage, a human
supported/unsupported/partial label and a short justification. Compare top-1
and top-3 on the same fixed questions; report counts by rank and label. Do not
put an answer accuracy number on a resume until those labels exist.

The bundled `chexpert_questions.json` contains 10 source-checked questions.
`offline_probe.py` and `chunk_sweep.py` use cached MiniLM embeddings and fixed
character windows, avoiding app dependencies for a local diagnostic. Their
reports are **not** the app's RecursiveCharacterTextSplitter/Chroma results.
`paired_answer_probe.py` compares a local Ollama model at k=1/3/4 on three
questions. Its outputs require human grounding labels before answer-quality
rates can be reported. The app itself uses Claude and expands from top-3 to
top-4 only after an explicit abstention.

