from src.evaluation.faithfulness import evaluate_faithfulness


def test_scores_claims_that_appear_in_tool_evidence():
    result = evaluate_faithfulness(
        "Revenue was $8.7 billion and margin was 31.2%.",
        [{"tool": "search_financial_docs_tool", "output": "Revenue: $8.7 billion; margin: 31.2%."}],
    )
    assert result.score == 100
    assert result.checked_claims == 1


def test_flags_claims_not_found_in_tool_evidence():
    result = evaluate_faithfulness(
        "Revenue was $9.1 billion.",
        [{"tool": "search_financial_docs_tool", "output": "Revenue: $8.7 billion."}],
    )
    assert result.score == 0
    assert result.unsupported_claims == ["Revenue was $9.1 billion."]


def test_no_score_without_tool_evidence():
    result = evaluate_faithfulness("Revenue was $8.7 billion.", [])
    assert result.score is None
    assert not result.evidence_available
