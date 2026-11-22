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
reports are **not** the app's passage splitter and ChromaDB results.
`paired_answer_probe.py` compares a local Ollama model at k=1/3/4 on three
questions. Its outputs require human grounding labels before answer-quality
rates can be reported. The app expands from top-3 to top-4 only after an
explicit abstention.

## Current application pipeline

Install `requirements.txt`, start Ollama, and pull `qwen2.5:3b` for local answers.
The app and CLI use the same PDF extraction, overlapping passage splitter,
MiniLM embeddings and ChromaDB retrieval. Chroma telemetry is disabled in client
settings. The Streamlit app selects the local model when no `ANTHROPIC_API_KEY`
is configured. To use Claude, also install `langchain-anthropic` and set the key.

Run `streamlit run app.py` for the interface. For one answer, run
`python run_pipeline.py data/papers/chexpert.pdf --question "How many patients?"`.
For retrieval metrics and model answers on the question set, run
`python run_pipeline.py data/papers/chexpert.pdf --questions evaluations/chexpert_questions.json --output evaluations/full_results.json`.
Review each answer and quote before reporting factual accuracy. The quote check
confirms the source text exists but does not prove every answer claim follows.

