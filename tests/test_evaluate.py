from dataclasses import dataclass
from evaluate import evaluate_cases


@dataclass
class Doc:
    page_content: str
    metadata: dict


class Store:
    def similarity_search(self, query, k):
        return {
            "first": [Doc("target evidence", {"page": 1}), Doc("other", {"page": 2})],
            "second": [Doc("other", {"page": 2}), Doc("target evidence", {"page": 1})],
            "miss": [Doc("unrelated", {"page": 1})],
        }[query][:k]


def test_rank_metrics_and_missing_evidence():
    cases = [{"question": q, "page": 1, "evidence": "target evidence"}
             for q in ["first", "second", "miss"]]
    report = evaluate_cases(Store(), cases)
    assert report["recall_at_1"] == 1 / 3
    assert report["recall_at_3"] == 2 / 3
    assert report["mrr_at_3"] == .5
    assert [x["evidence_rank"] for x in report["details"]] == [1, 2, None]

