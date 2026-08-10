import os
import json
import time
from typing import List, Any, Optional
import pandas as pd
from evaluation.models import EvaluationCase, EvaluationResult, PlannerScores, GroundingScores
from evaluation.scorers import score_plan, verify_execution, score_report, diagnose_execution
from app.services.agent_service import DataInsightAgent
from app.services.dataset_inspector import DatasetInspector
from app.models.analysis import ProviderReport
from app.core.exceptions import PlanValidationError, GroundingValidationError, ProviderError
from app.config import settings

def _sanitize_error_message(msg: str) -> str:
    if not msg:
        return "Unknown error"
    msg = msg[:200]
    msg_lower = msg.lower()
    suspicious = ["ignore", "system", "instruction", "select", "drop", "union", "delete", "insert", "secret", "whoami", "os.system"]
    if any(s in msg_lower for s in suspicious):
        return "Sanitized execution error due to security policy"
    return msg

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
    failure_stage = None
    exception_type = None
    safe_error_detail = None
    rep_attempted = False
    rep_succeeded = False

    summary = DatasetInspector.inspect(df)
    agent = DataInsightAgent()

    # Pre-declare variables for phase preservation
    selected_plan = None
    results_list = None
    report_obj = None

    planner_scores = None
    execution_passed = False
    grounding_scores = None

    try:
        response = await agent.analyze(df, case.question)

        selected_plan = response.analysis_plan
        results_list = response.analysis_results
        report_obj = ProviderReport(
            findings=response.findings,
            limitations=response.limitations,
            recommendations=response.recommendations
        )
        rep_attempted = response.report_repair_attempted
        rep_succeeded = response.report_repair_succeeded

    except PlanValidationError as exc:
        error_cat = "plan_validation_failed"
        failure_stage = getattr(exc, "failure_stage", "plan_validation")
        exception_type = exc.__class__.__name__
        safe_error_detail = _sanitize_error_message(str(exc))
        selected_plan = getattr(exc, "invalid_plan", None)
        results_list = getattr(exc, "analysis_results", None)
        rep_attempted = getattr(exc, "report_repair_attempted", False)
        rep_succeeded = getattr(exc, "report_repair_succeeded", False)

    except GroundingValidationError as exc:
        error_cat = "grounding_validation_failed"
        failure_stage = getattr(exc, "failure_stage", "grounding_validation")
        exception_type = exc.__class__.__name__
        safe_error_detail = _sanitize_error_message(str(exc))
        selected_plan = getattr(exc, "analysis_plan", None)
        results_list = getattr(exc, "analysis_results", None)
        report_obj = getattr(exc, "report", None)
        rep_attempted = getattr(exc, "report_repair_attempted", False)
        rep_succeeded = getattr(exc, "report_repair_succeeded", False)

    except ProviderError as exc:
        error_cat = "provider_error"
        failure_stage = getattr(exc, "failure_stage", "report_generation")
        exception_type = exc.__class__.__name__
        safe_error_detail = _sanitize_error_message(str(exc))
        selected_plan = getattr(exc, "analysis_plan", None)
        results_list = getattr(exc, "analysis_results", None)
        rep_attempted = getattr(exc, "report_repair_attempted", False)
        rep_succeeded = getattr(exc, "report_repair_succeeded", False)

    except Exception as exc:
        error_cat = "unknown_error"
        failure_stage = getattr(exc, "failure_stage", "planning")
        exception_type = exc.__class__.__name__
        safe_error_detail = _sanitize_error_message(str(exc))
        selected_plan = getattr(exc, "analysis_plan", None)
        results_list = getattr(exc, "analysis_results", None)
        rep_attempted = getattr(exc, "report_repair_attempted", False)
        rep_succeeded = getattr(exc, "report_repair_succeeded", False)

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Phase-Preservation Scoring Logic:
    # 1. Score proposed AnalysisPlan
    if selected_plan:
        try:
            planner_scores = score_plan(selected_plan, case, summary)
            # Apply repair flags from agent state
            planner_scores.plan_repair_attempted = agent.plan_repair_attempted
            planner_scores.plan_repair_succeeded = agent.plan_repair_succeeded
        except Exception:
            pass

    if not planner_scores:
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

    # 2. Score execution
    if results_list and error_cat not in ("plan_validation_failed", "unknown_error", "provider_error"):
        if failure_stage != "execution":
            try:
                execution_passed = verify_execution(results_list, case)
            except Exception:
                execution_passed = False

    # 3. Score findings/grounding
    if report_obj and results_list:
        try:
            grounding_scores = score_report(report_obj, results_list, case)
            grounding_scores.report_repair_attempted = rep_attempted
            grounding_scores.report_repair_succeeded = rep_succeeded
        except Exception:
            pass

    if not grounding_scores:
        grounding_scores = GroundingScores(
            structurally_grounded=False,
            unsupported_numeric_claim_flags=0,
            causal_claim_flags=0,
            report_repair_attempted=rep_attempted,
            report_repair_succeeded=rep_succeeded
        )

    # 4. Final end-to-end success evaluation
    final_success = (
        error_cat is None
        and planner_scores.planner_success
        and execution_passed
        and grounding_scores.structurally_grounded
        and grounding_scores.causal_claim_flags == 0
        and grounding_scores.unsupported_numeric_claim_flags == 0
    )

    execution_diagnostics = diagnose_execution(results_list, case)

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
        execution_diagnostics=execution_diagnostics,
        failure_stage=failure_stage,
        exception_type=exception_type,
        safe_error_detail=safe_error_detail
    )
