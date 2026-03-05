import unittest

from app.router import (
    PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR,
    PLAN_GRAPH_PLUS_VECTOR,
    PLAN_KEYWORD_PLUS_VECTOR,
    PLAN_VECTOR_ONLY,
    escalated_plan,
    low_confidence_reason,
    select_retrieval_plan,
)


class RouterSelectionTests(unittest.TestCase):
    def test_identifier_heavy_queries_route_keyword_plus_vector(self) -> None:
        queries = [
            "Where is CALL GailTable used?",
            "Trace MODULE_X updates in source",
            "Find CUSTOMER_RECORD write path",
            "Show error handling in file hazard.f90",
            "Where is AB06 coefficient loaded from .inc files?",
            "Locate call qkdist in deagg code",
            "Find references to HAZGRIDXNGA13L",
            "What modifies ARRAY_2048 in the model?",
            "Search for call getFEA and getABsub",
            "Which file defines deaggGRID.f behavior?",
        ]
        routes = [select_retrieval_plan(question=query, mode="chat") for query in queries]
        hit_rate = sum(1 for route in routes if route == PLAN_KEYWORD_PLUS_VECTOR) / len(routes)
        self.assertGreaterEqual(hit_rate, 0.8)

    def test_structure_queries_route_graph_plus_vector(self) -> None:
        queries = [
            "What is the entry point for this program?",
            "Show dependencies for the hazard run",
            "What calls the deaggregation routines?",
            "Give me the call chain of hazard execution",
            "Explain data flow across the pipeline",
            "What is the impact of changing interpolation logic?",
            "Estimate blast radius for this routine",
            "Map dependencies between runner scripts and compute modules",
            "Show call chain from startup to output",
            "Where is the entry point and dependency path documented?",
        ]
        routes = [select_retrieval_plan(question=query, mode="chat") for query in queries]
        hit_rate = sum(1 for route in routes if route == PLAN_GRAPH_PLUS_VECTOR) / len(routes)
        self.assertGreaterEqual(hit_rate, 0.8)

    def test_conceptual_queries_route_vector_only(self) -> None:
        queries = [
            "Explain what this model does at a high level",
            "Summarize the purpose of this repository",
            "How does the hazard workflow generally operate?",
            "What are the main assumptions in the calculation?",
            "Describe the reliability tradeoffs in this codebase",
            "Why does this system produce multiple output artifacts?",
            "Give a conceptual overview of the architecture",
            "What are common failure modes in the pipeline?",
            "How can I reason about maintainability here?",
            "Explain the overall design in plain language",
        ]
        routes = [select_retrieval_plan(question=query, mode="chat") for query in queries]
        hit_rate = sum(1 for route in routes if route == PLAN_VECTOR_ONLY) / len(routes)
        self.assertGreaterEqual(hit_rate, 0.8)

    def test_call_chain_phrase_prefers_structure_route(self) -> None:
        route = select_retrieval_plan(
            question="What is the entry point and call chain for the deaggregation workflow?",
            mode="chat",
        )
        self.assertEqual(route, PLAN_GRAPH_PLUS_VECTOR)


class RouterEscalationTests(unittest.TestCase):
    def test_vector_only_escalates_once(self) -> None:
        reason = low_confidence_reason([])
        self.assertEqual(reason, "no_matches")
        first = escalated_plan(PLAN_VECTOR_ONLY, did_escalate=False)
        second = escalated_plan(first or PLAN_VECTOR_ONLY, did_escalate=True)
        self.assertEqual(first, PLAN_KEYWORD_PLUS_VECTOR)
        self.assertIsNone(second)

    def test_graph_plus_vector_escalates_to_full_hybrid_once(self) -> None:
        citations = [
            {"file_path": "src/a.f", "line_start": 10, "line_end": 20, "score": 0.2},
            {"file_path": "src/b.f", "line_start": 30, "line_end": 40, "score": 0.18},
        ]
        reason = low_confidence_reason(citations)
        self.assertEqual(reason, "insufficient_unique_results")
        first = escalated_plan(PLAN_GRAPH_PLUS_VECTOR, did_escalate=False)
        second = escalated_plan(first or PLAN_GRAPH_PLUS_VECTOR, did_escalate=True)
        self.assertEqual(first, PLAN_GRAPH_PLUS_KEYWORD_PLUS_VECTOR)
        self.assertIsNone(second)

    def test_non_escalatable_plan_does_not_escalate(self) -> None:
        citations = [
            {"file_path": "src/a.f", "line_start": 10, "line_end": 20, "score": 0.9},
            {"file_path": "src/b.f", "line_start": 30, "line_end": 40, "score": 0.8},
            {"file_path": "src/c.f", "line_start": 50, "line_end": 60, "score": 0.7},
        ]
        reason = low_confidence_reason(citations)
        self.assertIsNone(reason)
        self.assertIsNone(escalated_plan(PLAN_KEYWORD_PLUS_VECTOR, did_escalate=False))


if __name__ == "__main__":
    unittest.main()
