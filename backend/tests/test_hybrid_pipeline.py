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

    def test_fallback_reason_reports_graph_runtime_unavailable(self) -> None:
        reason = main.classify_graph_fallback_reason(
            graph_payload={
                "errors": ["gitnexus_unavailable: [Errno 2] No such file or directory: 'npx'"],
                "raw_counts": {"processes": 0, "nodes": 0, "edges": 0, "files": 0},
            },
            signals={"structure_intent": True},
            low_conf_reason="graph_unavailable",
        )
        self.assertEqual(reason, "graph_runtime_unavailable")

    def test_fallback_reason_reports_graph_not_indexed_from_runtime_error(self) -> None:
        reason = main.classify_graph_fallback_reason(
            graph_payload={
                "errors": ["Error: No indexed repositories. Run: gitnexus analyze"],
                "raw_counts": {"processes": 0, "nodes": 0, "edges": 0, "files": 0},
            },
            signals={"structure_intent": True},
            low_conf_reason="graph_unavailable",
        )
        self.assertEqual(reason, "graph_not_indexed")

    def test_default_gitnexus_repo_never_empty(self) -> None:
        original_repo_root_path = main.repo_root_path
        original_default_repo = main.settings.gitnexus_default_repo
        original_namespace = main.settings.pinecone_namespace
        try:
            main.repo_root_path = lambda: Path("/")  # type: ignore[assignment]
            main.settings.gitnexus_default_repo = None
            main.settings.pinecone_namespace = "nshmp-main:v1"
            self.assertEqual(main.default_gitnexus_repo(), "nshmp-main")
        finally:
            main.repo_root_path = original_repo_root_path  # type: ignore[assignment]
            main.settings.gitnexus_default_repo = original_default_repo
            main.settings.pinecone_namespace = original_namespace

    def test_default_gitnexus_repo_prefers_namespace_over_generic_config(self) -> None:
        original_default_repo = main.settings.gitnexus_default_repo
        original_namespace = main.settings.pinecone_namespace
        try:
            main.settings.gitnexus_default_repo = "app"
            main.settings.pinecone_namespace = "nshmp-main:v1"
            self.assertEqual(main.default_gitnexus_repo(), "nshmp-main")
        finally:
            main.settings.gitnexus_default_repo = original_default_repo
            main.settings.pinecone_namespace = original_namespace

    def test_resolve_gitnexus_repo_name_prefers_namespace_when_generic_selected(self) -> None:
        original_namespace = main.settings.pinecone_namespace
        try:
            main.settings.pinecone_namespace = "nshmp-main:v1"
            resolved = main.resolve_gitnexus_repo_name("app", ["app", "nshmp-main"])
            self.assertEqual(resolved, "nshmp-main")
        finally:
            main.settings.pinecone_namespace = original_namespace

    def test_build_hybrid_graph_canvas_contains_nodes_and_edges(self) -> None:
        payload = main.build_hybrid_graph_canvas(
            question="What calls GailTable?",
            query_result={
                "processes": [{"id": "p1", "summary": "Main -> GailTable", "priority": 0.9}],
                "process_symbols": [{"id": "GailTable", "process_id": "p1", "filePath": "src/deaggGRID.f"}],
                "definitions": [{"id": "GailTable", "filePath": "src/deaggGRID.f"}],
            },
            context_result={"symbol": {"id": "GailTable", "filePath": "src/deaggGRID.f"}},
            impact_result={"byDepth": {"1": [{"id": "hazall", "filePath": "src/hazallXL.v5.f"}]}},
            candidate_ranking=[{"file_path": "src/deaggGRID.f", "score": 3.1}],
        )
        self.assertTrue(payload.get("nodes"))
        self.assertTrue(payload.get("edges"))

    def test_gitnexus_bootstrap_clone_url_injects_token(self) -> None:
        original_token = main.settings.gitnexus_bootstrap_git_token
        try:
            main.settings.gitnexus_bootstrap_git_token = "abc123"
            clone_url = main.gitnexus_bootstrap_clone_url("https://github.com/org/private-repo")
            self.assertIn("x-access-token:abc123@", clone_url)
        finally:
            main.settings.gitnexus_bootstrap_git_token = original_token

    def test_run_routed_hybrid_includes_graph_debug_when_graph_hits_exist(self) -> None:
        original_run_gitnexus_graph = main.run_gitnexus_graph
        original_retrieve = main.retrieve_with_optional_uploads
        try:
            main.run_gitnexus_graph = lambda question, repo_name=None, include_tests=False: {  # type: ignore[assignment]
                "repo": repo_name or "nshmp-main",
                "query": question,
                "processes": [{"id": "hazard_flow", "summary": "Hazard flow", "priority": 0.92}],
                "entrypoints": ["hazard_flow"],
                "impact": {},
                "candidate_files": ["src/hazallXL.v5.f"],
                "candidate_file_ranking": [{"file_path": "src/hazallXL.v5.f", "score": 3.2, "reasons": ["query.process_symbols"]}],
                "target_symbol": "hazard_flow",
                "errors": [],
                "raw_counts": {"processes": 1, "nodes": 6, "edges": 4, "files": 1},
                "score": {"best": 0.92, "threshold": 0.2, "passed": True},
                "index": {
                    "repo_id": repo_name or "nshmp-main",
                    "commit_hash": "abc1234",
                    "available_repos": [repo_name or "nshmp-main"],
                    "index_present": True,
                    "build_timestamp": "2026-03-05T00:00:00Z",
                    "node_count": 1200,
                    "edge_count": 3400,
                },
            }

            def _fake_retrieve(**_kwargs):  # type: ignore[no-untyped-def]
                citations = [
                    main.Citation(
                        file_path="src/hazallXL.v5.f",
                        line_start=100,
                        line_end=130,
                        score=0.86,
                        source_type="repo",
                        snippet="subroutine hazard_flow()",
                    ),
                    main.Citation(
                        file_path="src/hazpoint.f",
                        line_start=45,
                        line_end=70,
                        score=0.81,
                        source_type="repo",
                        snippet="subroutine hazpoint()",
                    ),
                    main.Citation(
                        file_path="src/hazinterp.f",
                        line_start=10,
                        line_end=32,
                        score=0.78,
                        source_type="repo",
                        snippet="subroutine hazinterp()",
                    ),
                ]
                chunks = [
                    "subroutine hazard_flow()\ncall compute()",
                    "subroutine hazpoint()\ncall interpolate()",
                    "subroutine hazinterp()\ncall finalize()",
                ]
                debug = {"index": {"timings_ms": {"pinecone_query": 20.0, "rerank": 4.0}}}
                return citations, chunks, debug

            main.retrieve_with_optional_uploads = _fake_retrieve  # type: ignore[assignment]

            citations, _, _, graph_payload, route_debug, _ = main.run_routed_retrieval_plan(
                question="What is the entry point and call chain for hazard flow?",
                mode="hybrid",
                top_k=5,
                scope="repo",
                project_id="nshmp-main",
            )
            self.assertTrue(citations)
            self.assertGreater(len(graph_payload.get("processes", [])), 0)
            hybrid_debug = route_debug.get("hybrid_debug", {})
            self.assertEqual(hybrid_debug.get("fallback_reason"), "")
            self.assertTrue(hybrid_debug.get("graph_index_present"))
            self.assertGreater(hybrid_debug.get("graph_hits", {}).get("processes", 0), 0)
        finally:
            main.run_gitnexus_graph = original_run_gitnexus_graph  # type: ignore[assignment]
            main.retrieve_with_optional_uploads = original_retrieve  # type: ignore[assignment]

    def test_hybrid_mode_always_executes_graph_step(self) -> None:
        original_run_gitnexus_graph = main.run_gitnexus_graph
        original_retrieve = main.retrieve_with_optional_uploads
        try:
            called = {"graph": 0}

            def _fake_graph(question, repo_name=None, include_tests=False):  # type: ignore[no-untyped-def]
                called["graph"] += 1
                return {
                    "repo": repo_name or "nshmp-main",
                    "query": question,
                    "processes": [],
                    "entrypoints": [],
                    "impact": {},
                    "candidate_files": [],
                    "candidate_file_ranking": [],
                    "target_symbol": None,
                    "errors": [],
                    "raw_counts": {"processes": 0, "nodes": 0, "edges": 0, "files": 0},
                    "score": {"best": 0.0, "threshold": 0.2, "passed": False},
                    "index": {
                        "repo_id": repo_name or "nshmp-main",
                        "commit_hash": "abc1234",
                        "available_repos": [repo_name or "nshmp-main"],
                        "index_present": True,
                        "build_timestamp": "2026-03-05T00:00:00Z",
                        "node_count": 1200,
                        "edge_count": 3400,
                    },
                }

            def _fake_retrieve(**_kwargs):  # type: ignore[no-untyped-def]
                return [], [], {"index": {"timings_ms": {"pinecone_query": 10.0, "rerank": 2.0}}}

            main.run_gitnexus_graph = _fake_graph  # type: ignore[assignment]
            main.retrieve_with_optional_uploads = _fake_retrieve  # type: ignore[assignment]

            _, _, _, _, route_debug, _ = main.run_routed_retrieval_plan(
                question="Explain this repository at a high level.",
                mode="hybrid",
                top_k=5,
                scope="repo",
                project_id="nshmp-main",
            )

            self.assertGreaterEqual(called["graph"], 1)
            self.assertEqual(route_debug.get("mode_override"), "hybrid_prefers_graph")
            self.assertTrue(any(step.get("name") == "graph" for step in route_debug.get("steps", [])))
        finally:
            main.run_gitnexus_graph = original_run_gitnexus_graph  # type: ignore[assignment]
            main.retrieve_with_optional_uploads = original_retrieve  # type: ignore[assignment]

    def test_architecture_query_excludes_tests_by_default(self) -> None:
        original_run_gitnexus_graph = main.run_gitnexus_graph
        original_retrieve = main.retrieve_with_optional_uploads
        try:
            main.run_gitnexus_graph = lambda question, repo_name=None, include_tests=False: {  # type: ignore[assignment]
                "repo": repo_name or "nshmp-main",
                "query": question,
                "processes": [{"id": "p1", "summary": "Main -> GailTable", "priority": 0.84}],
                "entrypoints": ["Main -> GailTable"],
                "impact": {},
                "candidate_files": ["backend/tests/test_router.py", "src/deaggGRID.f"],
                "candidate_file_ranking": [
                    {"file_path": "backend/tests/test_router.py", "score": 2.0, "reasons": ["query.process_symbols"]},
                    {"file_path": "src/deaggGRID.f", "score": 1.8, "reasons": ["query.process_symbols"]},
                ],
                "target_symbol": "gailtable",
                "errors": [],
                "raw_counts": {"processes": 1, "nodes": 4, "edges": 3, "files": 2},
                "score": {"best": 0.84, "threshold": 0.2, "passed": True},
                "index": {"repo_id": repo_name or "nshmp-main", "index_present": True},
            }

            def _fake_retrieve(**kwargs):  # type: ignore[no-untyped-def]
                include_tests = bool(kwargs.get("include_tests"))
                debug = {
                    "index": {"timings_ms": {"pinecone_query": 12.0, "rerank": 2.0}},
                    "test_filter": {
                        "include_tests": include_tests,
                        "excluded_test_candidates": 0,
                        "final_test_candidates": 0,
                    },
                }
                return (
                    [
                        main.Citation(
                            file_path="src/deaggGRID.f",
                            line_start=100,
                            line_end=130,
                            score=0.83,
                            source_type="repo",
                        )
                    ],
                    ["subroutine gailtable"],
                    debug,
                )

            main.retrieve_with_optional_uploads = _fake_retrieve  # type: ignore[assignment]

            _, _, _, graph_payload, route_debug, _ = main.run_routed_retrieval_plan(
                question="Show the call chain for GailTable including upstream callers and downstream impact.",
                mode="hybrid",
                top_k=5,
                scope="repo",
                project_id="nshmp-main",
            )

            hybrid_debug = route_debug.get("hybrid_debug", {})
            self.assertFalse(hybrid_debug.get("include_tests"))
            self.assertFalse(hybrid_debug.get("test_intent_detected"))
            self.assertFalse(hybrid_debug.get("fallback_with_tests"))
            self.assertGreater(hybrid_debug.get("excluded_test_candidates", 0), 0)
            self.assertNotIn("backend/tests/test_router.py", graph_payload.get("candidate_files", []))
        finally:
            main.run_gitnexus_graph = original_run_gitnexus_graph  # type: ignore[assignment]
            main.retrieve_with_optional_uploads = original_retrieve  # type: ignore[assignment]

    def test_test_query_enables_tests(self) -> None:
        original_run_gitnexus_graph = main.run_gitnexus_graph
        original_retrieve = main.retrieve_with_optional_uploads
        try:
            main.run_gitnexus_graph = lambda question, repo_name=None, include_tests=False: {  # type: ignore[assignment]
                "repo": repo_name or "nshmp-main",
                "query": question,
                "processes": [{"id": "p1", "summary": "Tests -> GailTable", "priority": 0.62}],
                "entrypoints": ["Tests -> GailTable"],
                "impact": {},
                "candidate_files": ["backend/tests/test_router.py", "src/deaggGRID.f"],
                "candidate_file_ranking": [],
                "target_symbol": "gailtable",
                "errors": [],
                "raw_counts": {"processes": 1, "nodes": 3, "edges": 2, "files": 2},
                "score": {"best": 0.62, "threshold": 0.2, "passed": True},
                "index": {"repo_id": repo_name or "nshmp-main", "index_present": True},
            }

            def _fake_retrieve(**kwargs):  # type: ignore[no-untyped-def]
                include_tests = bool(kwargs.get("include_tests"))
                file_path = "backend/tests/test_router.py" if include_tests else "src/deaggGRID.f"
                debug = {
                    "index": {"timings_ms": {"pinecone_query": 11.0, "rerank": 2.0}},
                    "test_filter": {
                        "include_tests": include_tests,
                        "excluded_test_candidates": 0,
                        "final_test_candidates": 1 if include_tests else 0,
                    },
                }
                return (
                    [
                        main.Citation(
                            file_path=file_path,
                            line_start=12,
                            line_end=33,
                            score=0.79,
                            source_type="repo",
                        )
                    ],
                    ["test coverage reference"],
                    debug,
                )

            main.retrieve_with_optional_uploads = _fake_retrieve  # type: ignore[assignment]

            _, _, _, _, route_debug, _ = main.run_routed_retrieval_plan(
                question="Which tests cover GailTable?",
                mode="hybrid",
                top_k=5,
                scope="repo",
                project_id="nshmp-main",
            )

            hybrid_debug = route_debug.get("hybrid_debug", {})
            self.assertTrue(hybrid_debug.get("include_tests"))
            self.assertTrue(hybrid_debug.get("test_intent_detected"))
            self.assertFalse(hybrid_debug.get("fallback_with_tests"))
        finally:
            main.run_gitnexus_graph = original_run_gitnexus_graph  # type: ignore[assignment]
            main.retrieve_with_optional_uploads = original_retrieve  # type: ignore[assignment]

    def test_low_confidence_triggers_single_fallback_with_tests(self) -> None:
        original_run_gitnexus_graph = main.run_gitnexus_graph
        original_retrieve = main.retrieve_with_optional_uploads
        try:
            main.run_gitnexus_graph = lambda question, repo_name=None, include_tests=False: {  # type: ignore[assignment]
                "repo": repo_name or "nshmp-main",
                "query": question,
                "processes": [{"id": "p1", "summary": "Main -> Unknown", "priority": 0.41}],
                "entrypoints": ["Main -> Unknown"],
                "impact": {},
                "candidate_files": ["src/deaggGRID.f", "backend/tests/test_router.py"],
                "candidate_file_ranking": [],
                "target_symbol": "unknown",
                "errors": [],
                "raw_counts": {"processes": 1, "nodes": 2, "edges": 1, "files": 2},
                "score": {"best": 0.41, "threshold": 0.2, "passed": True},
                "index": {"repo_id": repo_name or "nshmp-main", "index_present": True},
            }

            calls = {"count": 0}

            def _fake_retrieve(**kwargs):  # type: ignore[no-untyped-def]
                calls["count"] += 1
                include_tests = bool(kwargs.get("include_tests"))
                debug = {
                    "index": {"timings_ms": {"pinecone_query": 9.0, "rerank": 1.5}},
                    "test_filter": {
                        "include_tests": include_tests,
                        "excluded_test_candidates": 1 if not include_tests else 0,
                        "final_test_candidates": 1 if include_tests else 0,
                    },
                }
                if not include_tests:
                    return [], [], debug
                return (
                    [
                        main.Citation(
                            file_path="backend/tests/test_router.py",
                            line_start=21,
                            line_end=44,
                            score=0.77,
                            source_type="repo",
                        )
                    ],
                    ["assert route debug"],
                    debug,
                )

            main.retrieve_with_optional_uploads = _fake_retrieve  # type: ignore[assignment]

            citations, _, _, _, route_debug, _ = main.run_routed_retrieval_plan(
                question="Show call chain for GailTable including upstream callers and downstream impact.",
                mode="hybrid",
                top_k=5,
                scope="repo",
                project_id="nshmp-main",
            )

            hybrid_debug = route_debug.get("hybrid_debug", {})
            self.assertTrue(route_debug.get("escalation", {}).get("did_escalate"))
            self.assertTrue(hybrid_debug.get("fallback_with_tests"))
            self.assertTrue(hybrid_debug.get("include_tests"))
            self.assertTrue(citations)
            self.assertGreaterEqual(calls["count"], 2)
        finally:
            main.run_gitnexus_graph = original_run_gitnexus_graph  # type: ignore[assignment]
            main.retrieve_with_optional_uploads = original_retrieve  # type: ignore[assignment]

    def test_compose_hybrid_answer_reports_graph_not_indexed(self) -> None:
        answer = main.compose_hybrid_answer(
            question="Show call chain for hazard flow",
            graph={
                "repo": "nshmp-main",
                "errors": ["gitnexus_repo_not_indexed:nshmp-main"],
                "processes": [],
                "entrypoints": [],
                "candidate_files": [],
                "impact": {},
                "hybrid_debug": {
                    "fallback_reason": "graph_not_indexed",
                    "graph_metadata": {"repo_id": "nshmp-main", "commit_hash": "abc1234"},
                },
            },
            evidence_rows=[],
            used_fallback=True,
        )
        self.assertIn("Graph not indexed for repo nshmp-main at commit abc1234", answer)
        self.assertIn("npx -y gitnexus@latest analyze", answer)


if __name__ == "__main__":
    unittest.main()
