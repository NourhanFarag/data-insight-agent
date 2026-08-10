import pytest
import os
import json
import asyncio
from pydantic import ValidationError
from unittest.mock import MagicMock, patch
from evaluation.models import EvaluationCase, ExpectedResultCheck, EvaluationResult, PlannerScores, GroundingScores
from evaluation.runner import load_cases, evaluate_case
from evaluation.scorers import score_plan, verify_execution, score_report
from evaluation.report import compile_metrics_markdown
from app.models.analysis import AnalysisPlan, AnalysisStep, AnalysisOperation, AnalysisResult, ProviderReport, Finding, Recommendation, RecommendationPriority, ConfidenceLevel
from app.models.responses import DatasetSummary
from app.core.exceptions import ProviderError

# 1. Case Loading Tests

def test_case_loading_valid():
    """Asserts that a valid case file loads correctly."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cases_dir = os.path.join(base_dir, "..", "evaluation", "cases")
    cases = load_cases(cases_dir)
    assert len(cases) >= 10
    for case in cases:
        assert isinstance(case, EvaluationCase)
        assert os.path.exists(case.dataset_path)

def test_case_loading_missing_dataset():
    """Asserts that if a case references a missing dataset file, it is rejected."""
    case_data = {
      "case_id": "missing_dataset_test",
      "dataset_path": "nonexistent_file.csv",
      "question": "What is overall average?",
      "required_operations": ["MEAN"],
      "expected_result_checks": []
    }

    # Save a temporary JSON file to cases dir or a temp path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        case_file = os.path.join(tmpdir, "test_case.json")
        with open(case_file, "w", encoding="utf-8") as f:
            json.dump(case_data, f)

        with pytest.raises(FileNotFoundError):
            load_cases(tmpdir)

def test_case_loading_malformed_operation():
    """Asserts that a malformed operation throws a Pydantic validation error."""
    case_data = {
      "case_id": "malformed_op_test",
      "dataset_path": "evaluation/datasets/sales_basic.csv",
      "question": "What is overall average?",
      "required_operations": ["INVALID_OP"],  # Malformed enum
      "expected_result_checks": []
    }

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        case_file = os.path.join(tmpdir, "test_case.json")
        with open(case_file, "w", encoding="utf-8") as f:
            json.dump(case_data, f)

        with pytest.raises(ValueError) as exc:
            load_cases(tmpdir)
        assert "validation error" in str(exc.value).lower()


# 2. Plan Scoring Tests

@pytest.fixture
def mock_summary():
    return DatasetSummary(
        row_count=100, column_count=2, column_names=["revenue", "department"],
        inferred_data_types={"revenue": "numeric", "department": "categorical"},
        missing_value_count={"revenue": 0, "department": 0},
        numeric_columns=["revenue"], categorical_columns=["department"]
    )

def test_planner_scoring_perfect(mock_summary):
    proposed = AnalysisPlan(
        objective="Analyze rows",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.COUNT),
            AnalysisStep(step_id="step_2", operation=AnalysisOperation.MEAN, column="revenue")
        ]
    )

    case = EvaluationCase(
        case_id="test", dataset_path="evaluation/datasets/sales_basic.csv", question="Q",
        required_operations=[AnalysisOperation.MEAN],
        acceptable_operations=[AnalysisOperation.COUNT],
        expected_result_checks=[]
    )

    scores = score_plan(proposed, case, mock_summary)
    assert scores.schema_valid is True
    assert scores.plan_valid is True
    assert scores.required_operation_recall == 1.0
    assert scores.irrelevant_operation_rate == 0.0
    assert scores.invalid_column_attempts == 0
    assert scores.planner_success is True

def test_planner_scoring_zero_required_operations(mock_summary):
    proposed = AnalysisPlan(
        objective="Analyze rows",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.COUNT)
        ]
    )

    case = EvaluationCase(
        case_id="test", dataset_path="evaluation/datasets/sales_basic.csv", question="Q",
        required_operations=[],
        acceptable_operations=[AnalysisOperation.COUNT],
        expected_result_checks=[]
    )

    scores = score_plan(proposed, case, mock_summary)
    assert scores.required_operation_recall == 1.0
    assert scores.planner_success is True

def test_planner_scoring_missing_required(mock_summary):
    proposed = AnalysisPlan(
        objective="Analyze rows",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.COUNT)
        ]
    )

    case = EvaluationCase(
        case_id="test", dataset_path="evaluation/datasets/sales_basic.csv", question="Q",
        required_operations=[AnalysisOperation.MEAN],
        acceptable_operations=[AnalysisOperation.COUNT],
        expected_result_checks=[]
    )

    scores = score_plan(proposed, case, mock_summary)
    assert scores.required_operation_recall == 0.0
    assert scores.planner_success is False

def test_planner_scoring_irrelevant_steps(mock_summary):
    proposed = AnalysisPlan(
        objective="Analyze rows",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEAN, column="revenue"),
            AnalysisStep(step_id="step_2", operation=AnalysisOperation.CORRELATION, column="revenue", second_column="revenue")
        ]
    )

    case = EvaluationCase(
        case_id="test", dataset_path="evaluation/datasets/sales_basic.csv", question="Q",
        required_operations=[AnalysisOperation.MEAN],
        acceptable_operations=[],
        expected_result_checks=[]
    )

    scores = score_plan(proposed, case, mock_summary)
    # Correlation is irrelevant in this case
    assert scores.irrelevant_operation_rate == 0.5


# 3. Result Checking Tests

def test_result_checking_scalar_match():
    results = [
        AnalysisResult(
            result_id="result_1", source_step_id="step_1", operation=AnalysisOperation.MEAN,
            target_columns=["revenue"], computed_result=300.0, description="Mean is 300"
        )
    ]

    case = EvaluationCase(
        case_id="test", dataset_path="evaluation/datasets/sales_basic.csv", question="Q",
        required_operations=[],
        expected_result_checks=[
            ExpectedResultCheck(operation=AnalysisOperation.MEAN, column="revenue", expected_value=300.0, tolerance=0.01)
        ]
    )

    assert verify_execution(results, case) is True

    # Mismatch value
    case_fail = EvaluationCase(
        case_id="test", dataset_path="evaluation/datasets/sales_basic.csv", question="Q",
        required_operations=[],
        expected_result_checks=[
            ExpectedResultCheck(operation=AnalysisOperation.MEAN, column="revenue", expected_value=400.0)
        ]
    )
    assert verify_execution(results, case_fail) is False

def test_result_checking_grouped_match():
    results = [
        AnalysisResult(
            result_id="result_1", source_step_id="step_1", operation=AnalysisOperation.GROUP_BY_MEAN,
            target_columns=["revenue"], grouping_column="department",
            computed_result={"Engineering": 500.0, "Sales": 200.0}, description="Group mean"
        )
    ]

    case = EvaluationCase(
        case_id="test", dataset_path="evaluation/datasets/sales_basic.csv", question="Q",
        required_operations=[],
        expected_result_checks=[
            ExpectedResultCheck(
                operation=AnalysisOperation.GROUP_BY_MEAN, column="revenue", group_by="department",
                expected_value={"Engineering": 500.0, "Sales": 200.0}, tolerance=0.01
            )
        ]
    )

    assert verify_execution(results, case) is True


# 4. Grounding and Text Scoring Tests

def test_grounding_scoring_valid():
    results = [
        AnalysisResult(
            result_id="result_1", source_step_id="step_1", operation=AnalysisOperation.MEAN,
            target_columns=["revenue"], computed_result=300.0, description="Mean is 300"
        )
    ]

    report = ProviderReport(
        findings=[
            Finding(id="finding_1", title="Mean", explanation="The computed average is 300.0", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH)
        ],
        limitations=["Some limitation"],
        recommendations=[
            Recommendation(id="recommendation_1", priority=RecommendationPriority.MEDIUM, action="Act", rationale="Because of finding_1", finding_refs=["finding_1"])
        ]
    )

    case = EvaluationCase(
        case_id="test", dataset_path="evaluation/datasets/sales_basic.csv", question="Q",
        required_operations=[], expected_result_checks=[], tags=["correlation"]
    )

    scores = score_report(report, results, case)
    assert scores.structurally_grounded is True
    assert scores.unsupported_numeric_claim_flags == 0
    assert scores.causal_claim_flags == 0

def test_grounding_scoring_causal_flag():
    results = [
        AnalysisResult(
            result_id="result_1", source_step_id="step_1", operation=AnalysisOperation.CORRELATION,
            target_columns=["age", "spending"], computed_result=1.0, description="Corr is 1.0"
        )
    ]

    # Finding says: age causes spending (using prohibited causal keywords)
    report = ProviderReport(
        findings=[
            Finding(id="finding_1", title="Causal", explanation="Higher age drives/leads to spending increase because of age.", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH)
        ],
        limitations=[],
        recommendations=[]
    )

    case = EvaluationCase(
        case_id="test", dataset_path="evaluation/datasets/sales_basic.csv", question="Q",
        required_operations=[], expected_result_checks=[], tags=["correlation"]
    )

    scores = score_report(report, results, case)
    assert scores.causal_claim_flags > 0

def test_grounding_scoring_unsupported_numeric_flag():
    results = [
        AnalysisResult(
            result_id="result_1", source_step_id="step_1", operation=AnalysisOperation.MEAN,
            target_columns=["revenue"], computed_result=300.0, description="Mean is 300"
        )
    ]

    # Finding references 999.0 which was never computed
    report = ProviderReport(
        findings=[
            Finding(id="finding_1", title="Fabricated", explanation="The baseline metric is 999.0.", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH)
        ],
        limitations=[],
        recommendations=[]
    )

    case = EvaluationCase(
        case_id="test", dataset_path="evaluation/datasets/sales_basic.csv", question="Q",
        required_operations=[], expected_result_checks=[], tags=["basic"]
    )

    scores = score_report(report, results, case)
    assert scores.unsupported_numeric_claim_flags == 1


# 5. Run Resilience Tests

@patch("evaluation.runner.DataInsightAgent")
def test_run_resilience(mock_agent_class):
    """Verifies that an error in one case does not terminate the runner loop."""
    mock_agent = MagicMock()
    mock_agent_class.return_value = mock_agent

    # Mock behavior to raise exception
    mock_agent.analyze.side_effect = Exception("System Failure")

    case = EvaluationCase(
        case_id="resilience_case", dataset_path="evaluation/datasets/sales_basic.csv", question="Q",
        required_operations=[], expected_result_checks=[]
    )

    # Run evaluation case
    res = asyncio.run(evaluate_case(case, "mock"))
    assert res.case_id == "resilience_case"
    assert res.final_success is False
    assert res.error_category == "unknown_error"


# 6. Aggregation and Markdown Rendering Tests

def test_metrics_markdown_rendering():
    results = [
        EvaluationResult(
            case_id="case_1", provider="mock", model="mock",
            planner_scores=PlannerScores(schema_valid=True, plan_valid=True, required_operation_recall=1.0, irrelevant_operation_rate=0.0, invalid_column_attempts=0, planner_success=True),
            execution_passed=True,
            grounding_scores=GroundingScores(structurally_grounded=True, unsupported_numeric_claim_flags=0, causal_claim_flags=0),
            latency_ms=150.0, final_success=True
        ),
        EvaluationResult(
            case_id="case_2", provider="mock", model="mock",
            planner_scores=PlannerScores(schema_valid=True, plan_valid=True, required_operation_recall=0.0, irrelevant_operation_rate=1.0, invalid_column_attempts=0, planner_success=False),
            execution_passed=False,
            grounding_scores=GroundingScores(structurally_grounded=False, unsupported_numeric_claim_flags=1, causal_claim_flags=1),
            latency_ms=250.0, final_success=False
        )
    ]

    md_report = compile_metrics_markdown(results, "mock", 1)
    assert "# Data Insight Agent Evaluation Report" in md_report
    assert "harness verification only" in md_report.lower()
    assert "case_1" in md_report
    assert "case_2" in md_report
    assert "50.0%" in md_report  # success rate of 1/2


def test_evaluation_rate_bounds_and_exact_math():
    """Asserts that compile_metrics_markdown rates are strictly bounded [0.0, 1.0] even with anomalies."""
    results = [
        # Set values that could potentially break division boundaries
        EvaluationResult(
            case_id="case_1", provider="mock", model="mock",
            planner_scores=PlannerScores(schema_valid=True, plan_valid=True, required_operation_recall=1.0, irrelevant_operation_rate=0.0, invalid_column_attempts=0, planner_success=True),
            execution_passed=True,
            grounding_scores=GroundingScores(structurally_grounded=True, unsupported_numeric_claim_flags=0, causal_claim_flags=0),
            latency_ms=100.0, final_success=True
        ),
        EvaluationResult(
            case_id="case_1", provider="mock", model="mock",
            planner_scores=PlannerScores(schema_valid=True, plan_valid=True, required_operation_recall=1.0, irrelevant_operation_rate=0.0, invalid_column_attempts=0, planner_success=True),
            execution_passed=True,
            grounding_scores=GroundingScores(structurally_grounded=True, unsupported_numeric_claim_flags=0, causal_claim_flags=0),
            latency_ms=100.0, final_success=True
        )
    ]

    md_report = compile_metrics_markdown(results, "mock", 2)
    # Check that rates do not exceed 100%
    assert "100.5%" not in md_report
    assert "100.0%" in md_report
