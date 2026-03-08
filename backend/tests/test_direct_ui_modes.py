import unittest

from app import main


class DirectUiModeTests(unittest.TestCase):
    def test_execute_query_request_routes_audit_to_direct_agent(self) -> None:
        original_direct = main.execute_direct_ui_mode_request
        original_routed = main.run_routed_retrieval_plan
        try:
            called = {"direct": 0, "routed": 0}

            def _fake_direct(payload, telemetry, uploaded_files=None):  # type: ignore[no-untyped-def]
                called["direct"] += 1
                return main.QueryResponse(answer="audit ok", citations=[], telemetry={"mode": "audit"})

            def _fail_routed(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                called["routed"] += 1
                raise AssertionError("retrieval plan should not run for audit mode")

            main.execute_direct_ui_mode_request = _fake_direct  # type: ignore[assignment]
            main.run_routed_retrieval_plan = _fail_routed  # type: ignore[assignment]

            response = main.execute_query_request(
                main.QueryRequest(
                    question="Run a full repository audit.",
                    mode="chat",
                    ui_mode="audit",
                )
            )

            self.assertEqual(response.answer, "audit ok")
            self.assertEqual(called["direct"], 1)
            self.assertEqual(called["routed"], 0)
        finally:
            main.execute_direct_ui_mode_request = original_direct  # type: ignore[assignment]
            main.run_routed_retrieval_plan = original_routed  # type: ignore[assignment]

    def test_execute_query_request_routes_diagrams_to_direct_agent(self) -> None:
        original_direct = main.execute_direct_ui_mode_request
        original_routed = main.run_routed_retrieval_plan
        try:
            called = {"direct": 0, "routed": 0}

            def _fake_direct(payload, telemetry, uploaded_files=None):  # type: ignore[no-untyped-def]
                called["direct"] += 1
                return main.QueryResponse(answer="flowchart TD\nA-->B", citations=[], telemetry={"mode": "diagrams"})

            def _fail_routed(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                called["routed"] += 1
                raise AssertionError("retrieval plan should not run for diagrams mode")

            main.execute_direct_ui_mode_request = _fake_direct  # type: ignore[assignment]
            main.run_routed_retrieval_plan = _fail_routed  # type: ignore[assignment]

            response = main.execute_query_request(
                main.QueryRequest(
                    question="Draw the execution pipeline.",
                    mode="chat",
                    ui_mode="diagrams",
                    diagram_type="executionPipeline",
                )
            )

            self.assertIn("flowchart TD", response.answer)
            self.assertEqual(called["direct"], 1)
            self.assertEqual(called["routed"], 0)
        finally:
            main.execute_direct_ui_mode_request = original_direct  # type: ignore[assignment]
            main.run_routed_retrieval_plan = original_routed  # type: ignore[assignment]

    def test_build_direct_diagram_system_prompt_enforces_json_contract(self) -> None:
        prompt = main.build_direct_diagram_system_prompt("dataFlow")
        self.assertIn("Return only JSON", prompt)
        self.assertIn('"lanes"', prompt)
        self.assertIn("swimlanes", prompt)

    def test_build_mermaid_from_direct_diagram_spec_uses_subgraphs(self) -> None:
        spec = main.normalize_direct_diagram_spec(
            {
                "title": "National Hazard Run",
                "orientation": "LR",
                "lanes": [
                    {"id": "orchestration", "label": "Orchestration"},
                    {"id": "execution", "label": "Execution"},
                    {"id": "outputs", "label": "Outputs"},
                ],
                "nodes": [
                    {"id": "run_all", "label": "run_all_hazard.sh", "lane": "orchestration"},
                    {"id": "hazrun", "label": "hazrun_casc_2014.sh", "lane": "execution"},
                    {"id": "logs", "label": "logs/", "lane": "outputs"},
                ],
                "edges": [
                    {"from": "run_all", "to": "hazrun", "label": "executes"},
                    {"from": "hazrun", "to": "logs", "label": "writes"},
                ],
            },
            "executionPipeline",
        )
        mermaid = main.build_mermaid_from_direct_diagram_spec(spec)
        self.assertIn("flowchart LR", mermaid)
        self.assertIn('subgraph lane_orchestration["Orchestration"]', mermaid)
        self.assertIn('node_run_all["run_all_hazard.sh"]', mermaid)
        self.assertIn("node_run_all -->|executes| node_hazrun", mermaid)

    def test_normalize_direct_diagram_output_strips_fences(self) -> None:
        normalized = main.normalize_direct_diagram_output("```mermaid\nflowchart TD\nA[Start] --> B[End]\n```")
        self.assertEqual(normalized, "flowchart TD\nA[Start] --> B[End]")

    def test_normalize_direct_diagram_output_falls_back_for_invalid_text(self) -> None:
        normalized = main.normalize_direct_diagram_output("Here is your diagram:\nA -> B")
        self.assertEqual(normalized, "flowchart TD\nA[Diagram generation failed]")

    def test_execute_direct_ui_mode_request_returns_diagram_payload(self) -> None:
        original_context = main.build_direct_mode_context
        original_answer = main.generate_direct_mode_answer
        original_finalize = main.finalize_and_persist_telemetry
        try:
            main.build_direct_mode_context = lambda **_kwargs: (  # type: ignore[assignment]
                "repo context",
                [],
                {"selected_sources": []},
            )
            main.generate_direct_mode_answer = lambda **_kwargs: "flowchart TD\nA[Start] --> B[End]"  # type: ignore[assignment]
            main.finalize_and_persist_telemetry = lambda _telemetry: {"mode": "diagrams"}  # type: ignore[assignment]

            response = main.execute_direct_ui_mode_request(
                main.QueryRequest(
                    question="Draw the retrieval flow.",
                    mode="chat",
                    ui_mode="diagrams",
                    diagram_type="retrievalFlow",
                ),
                telemetry=main.build_request_telemetry(
                    main.QueryRequest(
                        question="Draw the retrieval flow.",
                        mode="chat",
                        ui_mode="diagrams",
                        diagram_type="retrievalFlow",
                    )
                ),
                uploaded_files=[],
            )

            self.assertEqual(response.type, "diagram")
            self.assertEqual(response.format, "mermaid")
            self.assertEqual(response.content, "flowchart TD\nA[Start] --> B[End]")
        finally:
            main.build_direct_mode_context = original_context  # type: ignore[assignment]
            main.generate_direct_mode_answer = original_answer  # type: ignore[assignment]
            main.finalize_and_persist_telemetry = original_finalize  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
