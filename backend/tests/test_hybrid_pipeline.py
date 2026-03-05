import unittest
from pathlib import Path

from app import main
from app.hybrid import extract_ranked_candidate_files, should_run_impact


class HybridPipelineTests(unittest.TestCase):
    def test_extract_ranked_candidate_files_merges_query_context_impact(self) -> None:
        query_result = {
            "processes": [{"id": "p1", "summary": "Hazard flow", "priority": 1.9}],
            "process_symbols": [
                {"id": "s1", "filePath": "src/hazallXL.v5.f", "process_id": "p1"},
                {"id": "s2", "filePath": "src/hazallXL.v4.f", "process_id": "p1"},
            ],
            "definitions": [{"id": "d1", "filePath": "conf/wus.conf"}],
        }
        context_result = {
            "symbol": {"uid": "s1", "filePath": "src/hazallXL.v5.f"},
            "incoming": {"calls": [{"uid": "c1", "filePath": "scripts/run_all_hazard.sh"}]},
            "outgoing": {"imports": [{"uid": "m1", "filePath": "conf/wus.conf"}]},
        }
        impact_result = {
            "target": {"id": "s1", "filePath": "src/hazallXL.v5.f"},
            "byDepth": {
                "1": [{"id": "x1", "filePath": "src/hazallXL.v5.f", "confidence": 1.0}],
                "2": [{"id": "x2", "filePath": "src/deaggGRID.f", "confidence": 0.8}],
            },
        }

        files, ranking = extract_ranked_candidate_files(
            query_result=query_result,
            context_result=context_result,
            impact_result=impact_result,
            max_candidate_files=10,
        )

        self.assertTrue(files)
        self.assertEqual(files[0], "src/hazallXL.v5.f")
        self.assertIn("src/deaggGRID.f", files)
        self.assertIn("scripts/run_all_hazard.sh", files)
        self.assertTrue(any(row.get("file_path") == "src/hazallXL.v5.f" for row in ranking))

    def test_should_run_impact_heuristics(self) -> None:
        self.assertTrue(should_run_impact("What breaks if I refactor hazallXL?"))
        self.assertTrue(should_run_impact("impact analysis for this module"))
        self.assertFalse(should_run_impact("where is hazallXL defined"))

    def test_build_pinecone_filter_supports_candidate_files(self) -> None:
        pinecone_filter = main.build_pinecone_filter(
            language="fortran",
            source_type="repo",
            file_paths=["src/hazallXL.v5.f", "src/deaggGRID.f"],
            repo="National-Seismic-Hazard-Maps",
        )
        self.assertIsInstance(pinecone_filter, dict)
        self.assertIn("$and", pinecone_filter)
        clauses = pinecone_filter["$and"]
        self.assertTrue(any("file_path" in clause and "$in" in clause["file_path"] for clause in clauses))
        self.assertTrue(any(clause.get("repo") == {"$eq": "National-Seismic-Hazard-Maps"} for clause in clauses))

    def test_infer_hybrid_target_prefers_called_symbol(self) -> None:
        question = "What calls GailTable, and what downstream routines does it impact?"
        target = main.infer_hybrid_target(question)
        self.assertEqual(str(target).lower(), "gailtable")

    def test_lexical_candidate_files_reports_rg_missing(self) -> None:
        original_run = main.subprocess.run
        original_repo_root_path = main.repo_root_path
        try:
            with main.lexical_candidate_cache_lock:
                main.lexical_candidate_cache.clear()

            def _raise_file_not_found(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                raise FileNotFoundError("rg not found")

            main.subprocess.run = _raise_file_not_found  # type: ignore[assignment]
            main.repo_root_path = lambda: Path(".")  # type: ignore[assignment]

            payload = main.lexical_candidate_files("What calls GailTable in this repo?")
            self.assertIn("rg_not_available", payload.get("errors", []))
        finally:
            main.subprocess.run = original_run  # type: ignore[assignment]
            main.repo_root_path = original_repo_root_path  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
