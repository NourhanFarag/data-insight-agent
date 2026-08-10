import pytest
import os
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from evaluation.models import EvaluationCase, ExpectedResultCheck, EvaluationResult, PlannerScores, GroundingScores, ExecutionCheckDiagnostic
from evaluation.runner import evaluate_case, resolve_model_name
from evaluation.scorers import diagnose_execution, _sanitize_actual_value
from evaluation.report import compile_metrics_markdown, save_evaluation_artifacts
from app.models.analysis import AnalysisPlan, AnalysisStep, AnalysisOperation, AnalysisResult
from app.config import settings

@pytest.fixture
def dummy_case():
    return EvaluationCase(
        case_id="adversarial_case",
        dataset_path="evaluation/datasets/sales_basic.csv",
        question="What is the average revenue?",
        required_operations=[AnalysisOperation.MEAN],
        expected_result_checks=[
            ExpectedResultCheck(
                operation=AnalysisOperation.MEAN,
                column="revenue",
                expected_value=250.0,
                tolerance=0.01
            )
        ]
    )

def test_resolve_model_name():
    """1. & 2. Test resolve_model_name helper logic."""
    with patch("app.config.settings.OLLAMA_MODEL", "qwen3:8b-test"), \
         patch("app.config.settings.GEMINI_MODEL", "gemini-3.5-test"), \
         patch("app.config.settings.OPENAI_MODEL", "gpt-test"):

        assert resolve_model_name("ollama") == "qwen3:8b-test"
        assert resolve_model_name("gemini") == "gemini-3.5-test"
        assert resolve_model_name("openai") == "gpt-test"
        assert resolve_model_name("mock") == "mock"
        assert resolve_model_name("unknown") == "mock"

def test_diagnose_execution_success():
    """3. Successful execution comparison diagnostics serialize correctly."""
    case = EvaluationCase(
        case_id="success_case",
        dataset_path="dummy.csv",
        question="Q",
        required_operations=[AnalysisOperation.MEAN],
        expected_result_checks=[
            ExpectedResultCheck(
                operation=AnalysisOperation.MEAN,
                column="revenue",
                expected_value=200.0,
                tolerance=0.1
            )
        ]
    )

    results = [
        AnalysisResult(
            result_id="result_1",
            source_step_id="step_1",
            operation=AnalysisOperation.MEAN,
            target_columns=["revenue"],
            computed_result=200.05,
            description="Mean is 200.05"
        )
    ]

    diags = diagnose_execution(results, case)
    assert len(diags) == 1
    assert diags[0].matching_result_found is True
    assert diags[0].comparison_outcome is True
    assert diags[0].mismatch_reason is None
    assert diags[0].actual_value == 200.05

def test_diagnose_execution_failures_and_precedence():
    """4. & 5. Failed execution diagnostics serialize correctly and check mismatch precedence."""
    case = EvaluationCase(
        case_id="fail_case",
        dataset_path="dummy.csv",
        question="Q",
        required_operations=[AnalysisOperation.MEAN, AnalysisOperation.TOP_VALUES],
        expected_result_checks=[
            ExpectedResultCheck(
                operation=AnalysisOperation.MEAN,
                column="revenue",
                expected_value=200.0,
                tolerance=0.01
            ),
            ExpectedResultCheck(
                operation=AnalysisOperation.TOP_VALUES,
                column="department",
                expected_value={"Sales": 2},
            )
        ]
    )

    # Precedence case 1: missing_expected_operation
    # Results has no MEAN operation at all
    results_missing_op = []
    diags_missing_op = diagnose_execution(results_missing_op, case)
    assert diags_missing_op[0].mismatch_reason == "missing_expected_operation"

    # Precedence case 2: column_mismatch
    # Results has MEAN but on orders, not revenue
    results_col_mismatch = [
        AnalysisResult(
            result_id="result_1",
            source_step_id="step_1",
            operation=AnalysisOperation.MEAN,
            target_columns=["orders"],
            computed_result=5.0,
            description="Mean is 5.0"
        )
    ]
    diags_col_mismatch = diagnose_execution(results_col_mismatch, case)
    assert diags_col_mismatch[0].mismatch_reason == "column_mismatch"

    # Precedence case 3: numeric_tolerance_mismatch
    # Results has MEAN on revenue, but value is 250.0 (expected 200.0, tol 0.01)
    results_val_mismatch = [
        AnalysisResult(
            result_id="result_1",
            source_step_id="step_1",
            operation=AnalysisOperation.MEAN,
            target_columns=["revenue"],
            computed_result=250.0,
            description="Mean is 250.0"
        )
    ]
    diags_val_mismatch = diagnose_execution(results_val_mismatch, case)
    assert diags_val_mismatch[0].mismatch_reason == "numeric_tolerance_mismatch"

def test_execution_not_reached():
    """6. execution_not_reached is used correctly when execution is not run."""
    case = EvaluationCase(
        case_id="not_reached_case",
        dataset_path="dummy.csv",
        question="Q",
        required_operations=[AnalysisOperation.MEAN],
        expected_result_checks=[
            ExpectedResultCheck(
                operation=AnalysisOperation.MEAN,
                column="revenue",
                expected_value=200.0,
            )
        ]
    )

    # Results is None represents execution not reached
    diags = diagnose_execution(None, case)
    assert len(diags) == 1
    assert diags[0].mismatch_reason == "execution_not_reached"
    assert diags[0].matching_result_found is False
    assert diags[0].comparison_outcome is False

def test_failed_markdown_diagnostics_render():
    """7. Failed Markdown diagnostics render correctly."""
    results = [
        EvaluationResult(
            case_id="case_fail", provider="ollama", model="qwen3:8b",
            planner_scores=PlannerScores(schema_valid=True, plan_valid=True, required_operation_recall=1.0, irrelevant_operation_rate=0.0, invalid_column_attempts=0, planner_success=True),
            execution_passed=False,
            grounding_scores=GroundingScores(structurally_grounded=True, unsupported_numeric_claim_flags=0, causal_claim_flags=0),
            latency_ms=150.0, final_success=False,
            selected_plan=AnalysisPlan(
                objective="Analyze department",
                steps=[
                    AnalysisStep(
                        step_id="step_1",
                        operation=AnalysisOperation.TOP_VALUES,
                        column="department",
                        reason="Get top department"
                    )
                ]
            ),
            execution_diagnostics=[
                ExecutionCheckDiagnostic(
                    expected_operation=AnalysisOperation.TOP_VALUES,
                    expected_column="department",
                    expected_value={"Sales": 10},
                    matching_result_found=True,
                    comparison_outcome=False,
                    mismatch_reason="value_mismatch",
                    actual_value="<redacted categorical value>"
                )
            ]
        )
    ]

    md_report = compile_metrics_markdown(results, "ollama", 1)

    assert "## Failed Case Diagnostics" in md_report
    assert "Case: case_fail" in md_report
    assert "Model: qwen3:8b" in md_report
    assert "Outcome: FAIL" in md_report
    assert "Mismatch: value_mismatch" in md_report
    assert "Actual: <redacted categorical value>" in md_report

def test_no_raw_csv_and_adversarial_safety(tmp_path):
    """8. Raw CSV rows/cells are not serialized.
       9. Adversarial payload does not leak into JSON.
       10. Adversarial payload does not leak into Markdown.
    """
    adversarial_payload = "'; DROP TABLE sales; --"

    results = [
        EvaluationResult(
            case_id="case_adversarial", provider="ollama", model="qwen3:8b",
            planner_scores=PlannerScores(schema_valid=True, plan_valid=True, required_operation_recall=1.0, irrelevant_operation_rate=0.0, invalid_column_attempts=0, planner_success=True),
            execution_passed=False,
            grounding_scores=GroundingScores(structurally_grounded=True, unsupported_numeric_claim_flags=0, causal_claim_flags=0),
            latency_ms=150.0, final_success=False,
            selected_plan=AnalysisPlan(
                objective="Adversarial objective",
                steps=[]
            ),
            execution_diagnostics=[
                ExecutionCheckDiagnostic(
                    expected_operation=AnalysisOperation.TOP_VALUES,
                    expected_column="department",
                    expected_value={"Sales": 10},
                    matching_result_found=True,
                    comparison_outcome=False,
                    mismatch_reason="value_mismatch",
                    actual_value=_sanitize_actual_value({"Sales": adversarial_payload}, {"Sales": 10}, False)
                )
            ]
        )
    ]

    json_path, md_path = save_evaluation_artifacts(results, "ollama", 1, output_dir=str(tmp_path))

    # 1. Verify JSON contents
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = f.read()
        # Verify the adversarial payload is redacted and does not appear in the file
        assert adversarial_payload not in json_data
        assert "<redacted categorical value>" in json_data

    # 2. Verify Markdown contents
    with open(md_path, "r", encoding="utf-8") as f:
        md_data = f.read()
        assert adversarial_payload not in md_data
        assert "<redacted" in md_data

def test_aggregate_metric_calculations_unchanged():
    """11. Aggregate metric calculations remain unchanged (e.g. recall, irrelevant rate, etc.)."""
    results = [
        EvaluationResult(
            case_id="case_1", provider="mock", model="mock",
            planner_scores=PlannerScores(schema_valid=True, plan_valid=True, required_operation_recall=0.8, irrelevant_operation_rate=0.2, invalid_column_attempts=0, planner_success=True),
            execution_passed=True,
            grounding_scores=GroundingScores(structurally_grounded=True, unsupported_numeric_claim_flags=0, causal_claim_flags=0),
            latency_ms=100.0, final_success=True
        )
    ]

    md_report = compile_metrics_markdown(results, "mock", 1)

    # Confirm math is exact
    assert "80.0%" in md_report  # recall
    assert "20.0%" in md_report  # irrelevant rate
    assert "100.0%" in md_report  # schema valid
