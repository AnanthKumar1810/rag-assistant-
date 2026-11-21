from dataclasses import dataclass
import pytest
from rag.structured import Answer, ask_structured


@dataclass
class Doc:
    page_content: str
    metadata: dict


class Store:
    def similarity_search(self, question, k):
        assert k == 3
        return [Doc("The dataset has 224,316 radiographs.", {"page": 2})]


class Model:
    def __init__(self, result): self.result = result
    def with_structured_output(self, schema, include_raw):
        assert schema is Answer and include_raw
        return self
    def invoke(self, prompt): return self.result


def test_supported_answer_and_source_page():
    answer, docs = ask_structured(Store(), "How many images?", Model({
        "parsed": Answer(answer="224,316 radiographs", source_pages=[2],
                         evidence_quotes=["224,316 radiographs"], found=True),
        "parsing_error": None}))
    assert answer.source_pages == [2] and docs[0].metadata["page"] == 2


@pytest.mark.parametrize("parsed,error", [
    (None, ValueError("invalid JSON")),
    (Answer(answer="claim", source_pages=[9], evidence_quotes=["claim"], found=True), None),
    (Answer(answer="claim", source_pages=[], evidence_quotes=[], found=False), None),
    (Answer(answer="claim", source_pages=[2], evidence_quotes=["invented quote"], found=True), None),
])
def test_invalid_json_or_source_fails_closed(parsed, error):
    with pytest.raises(ValueError):
        ask_structured(Store(), "question", Model({"parsed": parsed, "parsing_error": error}))

