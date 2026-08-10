import os
import json
import time
import pandas as pd
from typing import List
from evaluation.models import EvaluationCase, EvaluationResult, PlannerScores, GroundingScores
from evaluation.scorers import score_plan, verify_execution, score_report, diagnose_execution
from app.services.agent_service import DataInsightAgent
from app.models.analysis import ProviderReport
from app.core.exceptions import PlanValidationError, GroundingValidationError, ProviderError
from app.config import settings

def resolve_model_name(provider: str) -> str:
    """Centralized helper to resolve the model name for a provider."""
    if provider == "gemini":
        return settings.GEMINI_MODEL
    elif provider == "openai":
        return settings.OPENAI_MODEL
    elif provider == "ollama":
        return settings.OLLAMA_MODEL
    return "mock"

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
        planner_scores.plan_repair_attempted = response.plan_repair_attempted
        planner_scores.plan_repair_succeeded = response.plan_repair_succeeded

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

        selected_plan = response.analysis_plan
        execution_diagnostics = diagnose_execution(response.analysis_results, case)

    except PlanValidationError as exc:
        error_cat = "plan_validation_failed"
        selected_plan = getattr(exc, "invalid_plan", None)
        execution_diagnostics = diagnose_execution(None, case)
    except GroundingValidationError as exc:
        error_cat = "grounding_validation_failed"
        selected_plan = getattr(exc, "analysis_plan", None)
        results_list = getattr(exc, "analysis_results", None)
        execution_diagnostics = diagnose_execution(results_list, case)
    except ProviderError as exc:
        error_cat = "provider_error"
        selected_plan = getattr(exc, "analysis_plan", None)
        results_list = getattr(exc, "analysis_results", None)
        execution_diagnostics = diagnose_execution(results_list, case)
    except Exception as exc:
        error_cat = "unknown_error"
        selected_plan = getattr(exc, "analysis_plan", None)
        results_list = getattr(exc, "analysis_results", None)
        execution_diagnostics = diagnose_execution(results_list, case)

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Fallback score objects on exception
    if error_cat:
        planner_scores = PlannerScores(
            schema_valid=False,
            plan_valid=False,
            required_operation_recall=0.0,
            irrelevant_operation_rate=0.0,
            invalid_column_attempts=0,
            planner_success=False,
            plan_repair_attempted=agent.plan_repair_attempted,
            plan_repair_succeeded=agent.plan_repair_succeeded
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
        model=resolve_model_name(provider),
        planner_scores=planner_scores,
        execution_passed=execution_passed,
        grounding_scores=grounding_scores,
        latency_ms=latency_ms,
        error_category=error_cat,
        final_success=final_success,
        selected_plan=selected_plan,
        execution_diagnostics=execution_diagnostics
    )
