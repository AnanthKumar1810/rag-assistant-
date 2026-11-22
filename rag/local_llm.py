"""Ollama adapter for the same structured-answer contract used by Claude."""
import json
from urllib.request import Request, build_opener, ProxyHandler


class LocalOllama:
    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def with_structured_output(self, schema, include_raw=True):
        return _StructuredLocalModel(self.model, schema)


class _StructuredLocalModel:
    def __init__(self, model, schema):
        self.model, self.schema = model, schema

    def invoke(self, prompt):
        payload = {"model": self.model, "stream": False,
                   "format": self.schema.model_json_schema(),
                   "options": {"temperature": 0},
                   "messages": [{"role": "user", "content": prompt}]}
        request = Request("http://127.0.0.1:11434/api/chat",
                          data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        with build_opener(ProxyHandler({})).open(request, timeout=180) as response:
            raw = json.load(response)["message"]["content"]
        try:
            return {"raw": raw, "parsed": self.schema.model_validate_json(raw), "parsing_error": None}
        except (ValueError, TypeError) as exc:
            return {"raw": raw, "parsed": None, "parsing_error": exc}

