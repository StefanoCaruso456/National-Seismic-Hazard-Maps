import unittest

from app import main


class FocusTermGuardrailTests(unittest.TestCase):
    def test_extract_focus_terms_identifier_like_tokens(self) -> None:
        terms = main.extract_focus_terms("Explain `CALCULATE-INTEREST` and CUSTOMER-RECORD updates.")
        self.assertIn("calculate-interest", terms)
        self.assertIn("customer-record", terms)
        self.assertNotIn("end-to-end", terms)

    def test_analyze_focus_term_alignment_unmatched(self) -> None:
        citation = main.Citation(
            file_path="src/example.f",
            line_start=10,
            line_end=30,
            score=0.74,
            source_type="repo",
            snippet="subroutine hazallXL(iq, ia)",
        )
        alignment = main.analyze_focus_term_alignment(
            "What functions modify CUSTOMER-RECORD?",
            [citation],
            ["subroutine hazallXL(iq, ia)\ncall getFEA(...)"],
        )
        self.assertEqual(alignment["required_terms"], ["customer-record"])
        self.assertEqual(alignment["matched_terms"], [])
        self.assertEqual(alignment["unmatched_terms"], ["customer-record"])
        self.assertEqual(alignment["coverage"], 0.0)

    def test_compute_evidence_strength_caps_without_focus_match(self) -> None:
        citation = main.Citation(
            file_path="src/deaggGRID.f",
            line_start=100,
            line_end=120,
            score=0.82,
            source_type="repo",
            snippet="subroutine getFEA(iq, ir, ia, ndist, di, nmag, magmin, dmag)",
        )
        retrieval_debug = {
            "focus_guardrail": {
                "required_terms": ["customer-record"],
                "matched_terms": [],
                "unmatched_terms": ["customer-record"],
                "coverage": 0.0,
            },
            "index": {"subqueries": [{"query": "customer-record", "matches": 5}]},
            "uploads": {"subqueries": []},
        }
        evidence = main.compute_evidence_strength(
            "What functions modify CUSTOMER-RECORD?",
            [citation],
            retrieval_debug,
            mode="chat",
            mode_metrics={},
        )
        self.assertEqual(evidence["label"], "Low")
        self.assertLessEqual(evidence["score"], float(main.settings.retrieval_focus_term_absent_cap))
        self.assertIn("not found", evidence["reason"].lower())

    def test_compute_evidence_strength_allows_match(self) -> None:
        citation = main.Citation(
            file_path="src/customer_record.f",
            line_start=40,
            line_end=60,
            score=0.82,
            source_type="repo",
            snippet="subroutine update_customer_record(rec)\n! modifies CUSTOMER-RECORD",
        )
        retrieval_debug = {
            "focus_guardrail": {
                "required_terms": ["customer-record"],
                "matched_terms": ["customer-record"],
                "unmatched_terms": [],
                "coverage": 1.0,
            },
            "index": {"subqueries": [{"query": "customer-record", "matches": 5}]},
            "uploads": {"subqueries": []},
        }
        evidence = main.compute_evidence_strength(
            "What functions modify CUSTOMER-RECORD?",
            [citation],
            retrieval_debug,
            mode="chat",
            mode_metrics={},
        )
        self.assertGreater(evidence["score"], float(main.settings.retrieval_focus_term_absent_cap))


if __name__ == "__main__":
    unittest.main()
