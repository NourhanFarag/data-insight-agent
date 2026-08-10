import os
import json
import time
import pandas as pd
from typing import List
from evaluation.models import EvaluationCase, EvaluationResult, PlannerScores, GroundingScores
from evaluation.scorers import score_plan, verify_execution, score_report
from app.services.agent_service import DataInsightAgent
from app.models.analysis import ProviderReport
from app.core.exceptions import PlanValidationError, GroundingValidationError, ProviderError
from app.config import settings

def load_cases(cases_dir: str) -> List[EvaluationCase]:
    """Loads and validates all frozen evaluation cases from the cases directory."""
    cases = []
    if not os.path.exists(cases_dir):
        raise FileNotFoundError(f"Cases directory not found: {cases_dir}")

    for filename in sorted(os.listdir(cases_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(cases_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    case = EvaluationCase.model_validate(data)
                except Exception as e:
                    raise ValueError(f"Failed to parse or validate case file {filename}: {str(e)}")

                # Check dataset path existence outside parser try/except
                if not os.path.exists(case.dataset_path):
                    raise FileNotFoundError(f"Dataset path '{case.dataset_path}' not found for case {case.case_id}")
                cases.append(case)
    return cases

async def evaluate_case(case: EvaluationCase, provider: str) -> EvaluationResult:
    """Executes a single evaluation run for a case and scores the output."""
    # Ensure config matches requested provider
    settings.AI_PROVIDER = provider

    # Read the dataset
    df = pd.read_csv(case.dataset_path)

    start_time = time.perf_counter()
    error_cat = None
    planner_scores = None
    execution_passed = False
    grounding_scores = None
    final_success = False

    # Initialize the orchestrator agent service
    agent = DataInsightAgent()

    try:
        response = await agent.analyze(df, case.question)

        # 1. Score proposed AnalysisPlan
        planner_scores = score_plan(response.analysis_plan, case, response.dataset_summary)

        # 2. Verify deterministic calculations
        execution_passed = verify_execution(response.analysis_results, case)

        # 3. Score findings/grounding
        report_obj = ProviderReport(
            findings=response.findings,
            limitations=response.limitations,
            recommendations=response.recommendations
        )
        grounding_scores = score_report(report_obj, response.analysis_results, case)

        # 4. Final end-to-end success evaluation
        # Succeeded on plan AND calculation matches expected ground-truth AND report grounded with 0 flags
        final_success = (
            planner_scores.planner_success
            and execution_passed
            and grounding_scores.structurally_grounded
            and grounding_scores.causal_claim_flags == 0
            and grounding_scores.unsupported_numeric_claim_flags == 0
        )

    except PlanValidationError:
        error_cat = "plan_validation_failed"
    except GroundingValidationError:
        error_cat = "grounding_validation_failed"
    except ProviderError:
        error_cat = "provider_error"
    except Exception:
        error_cat = "unknown_error"

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Fallback score objects on exception
    if error_cat:
        planner_scores = PlannerScores(
            schema_valid=False,
            plan_valid=False,
            required_operation_recall=0.0,
            irrelevant_operation_rate=0.0,
            invalid_column_attempts=0,
            planner_success=False
        )
        grounding_scores = GroundingScores(
            structurally_grounded=False,
            unsupported_numeric_claim_flags=0,
            causal_claim_flags=0
        )
        execution_passed = False
        final_success = False

    return EvaluationResult(
        case_id=case.case_id,
        provider=provider,
        model=settings.GEMINI_MODEL if provider == "gemini" else (settings.OPENAI_MODEL if provider == "openai" else "mock"),
        planner_scores=planner_scores,
        execution_passed=execution_passed,
        grounding_scores=grounding_scores,
        latency_ms=latency_ms,
        error_category=error_cat,
        final_success=final_success
    )
