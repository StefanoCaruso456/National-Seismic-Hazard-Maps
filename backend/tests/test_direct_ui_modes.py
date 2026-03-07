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

    def test_build_direct_diagram_system_prompt_enforces_mermaid_only_contract(self) -> None:
        prompt = main.build_direct_diagram_system_prompt("dataFlow")
        self.assertIn("Return only valid Mermaid", prompt)
        self.assertIn("flowchart TD", prompt)
        self.assertIn("Use subgraph blocks as swimlanes", prompt)


if __name__ == "__main__":
    unittest.main()
