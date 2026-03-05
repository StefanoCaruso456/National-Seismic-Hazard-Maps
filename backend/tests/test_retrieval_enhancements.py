import unittest

from app import main
from app.ingest import line_is_comment, split_into_sections


class RetrievalEnhancementTests(unittest.TestCase):
    def test_fortran_section_parser_tracks_symbol_metadata(self) -> None:
        text = """
      module hazard_mod
      use geom_lib
      implicit none
      contains
      subroutine run_hazard(iq)
      call compute(iq)
      end subroutine run_hazard
      end module hazard_mod
""".strip(
            "\n"
        )
        sections = split_into_sections(text)
        self.assertTrue(sections)
        names = [section.symbol_name for section in sections if section.symbol_name]
        self.assertIn("hazard_mod", names)
        self.assertIn("run_hazard", names)
        module_section = next(section for section in sections if section.symbol_name == "hazard_mod")
        self.assertEqual(module_section.symbol_type, "module")
        self.assertTrue(module_section.contains_block)
        self.assertIn("geom_lib", module_section.imports)

    def test_identifier_hint_extraction(self) -> None:
        hints = main.extract_identifier_hints("Where does CALL GailTable update CUSTOMER_RECORD in HAZGRIDXNGA13L?")
        self.assertIn("gailtable", hints)
        self.assertIn("customer_record", hints)
        self.assertIn("hazgridxnga13l", hints)

    def test_identifier_hint_extraction_skips_plain_english(self) -> None:
        hints = main.extract_identifier_hints("Where is the main entry point of this program?")
        self.assertEqual(hints, [])

    def test_rerank_prefers_exact_identifier_and_symbol_match(self) -> None:
        matches = [
            {
                "score": 0.93,
                "metadata": {
                    "file_path": "src/noisy.f",
                    "line_start": 10,
                    "line_end": 30,
                    "chunk_text": "! comment\n! comment\nsubroutine helper\ncall work()",
                    "section_name": "helper",
                },
            },
            {
                "score": 0.72,
                "metadata": {
                    "file_path": "src/customer_record.f",
                    "line_start": 40,
                    "line_end": 70,
                    "chunk_text": "subroutine update_customer_record(rec)\ncall write(rec)",
                    "section_name": "update_customer_record",
                    "symbol_name": "update_customer_record",
                    "module_name": "customer_mod",
                    "_lexical_file_score": 1.0,
                },
            },
        ]
        reranked, debug = main.rerank_matches("What modifies CUSTOMER_RECORD?", matches, top_k=2)
        self.assertEqual(reranked[0][0].get("file_path"), "src/customer_record.f")
        self.assertGreaterEqual(len(debug), 1)
        self.assertEqual(debug[0]["file_path"], "src/customer_record.f")

    def test_expand_repo_context_uses_parent_definition_bounds(self) -> None:
        original_repo_file_lines = main.repo_file_lines
        original_find_enclosing_definition = main.find_enclosing_definition
        original_file_snippet = main.file_snippet
        try:
            mock_lines = tuple(
                [
                    "module m1",
                    "subroutine target()",
                    "call work()",
                    "call work2()",
                    "end subroutine target",
                    "end module m1",
                ]
            )
            main.repo_file_lines = lambda _path: mock_lines  # type: ignore[assignment]
            main.find_enclosing_definition = lambda _path, _line: {"line_start": 2, "line_end": 5}  # type: ignore[assignment]
            main.file_snippet = lambda _path, start, end, pad_after=0: "\n".join(mock_lines[start - 1 : end])  # type: ignore[assignment]

            citation = main.Citation(
                file_path="src/mock.f",
                line_start=3,
                line_end=3,
                score=0.8,
                source_type="repo",
                snippet="call work()",
            )
            expanded, expanded_text = main.expand_repo_citation_context("Explain target()", citation, "call work()")
            self.assertEqual(expanded.line_start, 2)
            self.assertEqual(expanded.line_end, 5)
            self.assertIn("subroutine target()", expanded_text.lower())
        finally:
            main.repo_file_lines = original_repo_file_lines  # type: ignore[assignment]
            main.find_enclosing_definition = original_find_enclosing_definition  # type: ignore[assignment]
            main.file_snippet = original_file_snippet  # type: ignore[assignment]

    def test_fortran_comment_detection_does_not_flag_call_statement(self) -> None:
        self.assertTrue(line_is_comment("c fixed-form comment"))
        self.assertTrue(line_is_comment("! free-form comment"))
        self.assertFalse(line_is_comment("      call work()"))


if __name__ == "__main__":
    unittest.main()
