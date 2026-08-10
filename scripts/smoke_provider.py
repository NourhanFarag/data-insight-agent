import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import asyncio
import json
import pandas as pd
from app.config import settings
from app.services.agent_service import DataInsightAgent
from app.core.exceptions import AppBaseException

async def run_smoke_test(provider_name: str):
    """Executes an end-to-end run of the orchestrator using the chosen provider."""
    print(f"--- Starting Smoke Test with Provider: {provider_name} ---")

    # Override settings
    settings.AI_PROVIDER = provider_name

    # Load env settings if available
    print(f"Current Configured Models:")
    print(f"  Gemini Model: {settings.GEMINI_MODEL}")
    print(f"  OpenAI Model: {settings.OPENAI_MODEL}")
    print(f"  Ollama Model: {settings.OLLAMA_MODEL} at {settings.OLLAMA_BASE_URL}")

    # Create simple deterministic dataset
    df = pd.DataFrame({
        "department": ["Sales", "Sales", "Engineering", "Engineering"],
        "revenue": [150.0, 250.0, 400.0, 600.0],
        "orders": [3, 5, 8, 12]
    })

    question = "Which department is performing best by average revenue and orders?"
    agent = DataInsightAgent()

    try:
        response = await agent.analyze(df, question)
        print("\n=== Agent Analysis Response ===")
        print(f"Question: {response.question}")
        print(f"Row count: {response.dataset_summary.row_count}")
        print(f"Column count: {response.dataset_summary.column_count}")
        print("\n--- Plan Steps ---")
        for step in response.analysis_plan.steps:
            print(f"  [{step.step_id}] {step.operation.value} on {step.column or 'all'} (Reason: {step.reason})")

        print("\n--- Executed Results ---")
        for res in response.analysis_results:
            print(f"  [{res.result_id}] Source step: {res.source_step_id} | {res.operation.value} -> {res.computed_result}")

        print("\n--- Findings ---")
        for f in response.findings:
            print(f"  [{f.id}] {f.title} (Evidence: {f.evidence_refs})")
            print(f"        Explanation: {f.explanation}")

        print("\n--- Limitations ---")
        for lim in response.limitations:
            print(f"  - {lim}")

        print("\n--- Recommendations ---")
        for r in response.recommendations:
            print(f"  [{r.id}] Priority: {r.priority.value} | {r.action} (Finding: {r.finding_refs})")
            print(f"        Rationale: {r.rationale}")

        print("\nSmoke test completed successfully!")

    except AppBaseException as e:
        print(f"\n[ERROR] Domain Exception Raised ({e.__class__.__name__}): {e.message}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run manual LLM provider smoke tests.")
    parser.add_argument(
        "--provider",
        choices=["mock", "ollama", "gemini", "openai"],
        default="mock",
        help="LLM provider to execute the run against (default: mock)."
    )
    args = parser.parse_args()

    asyncio.run(run_smoke_test(args.provider))
