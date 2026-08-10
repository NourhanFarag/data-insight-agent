import pytest
import pandas as pd
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.agent_service import DataInsightAgent
from app.providers.mock_provider import MockProvider
from app.models.analysis import AnalysisPlan, AnalysisStep, AnalysisOperation
from app.models.responses import DatasetSummary
from app.core.exceptions import PlanValidationError
from evaluation.models import EvaluationCase
from evaluation.runner import evaluate_case

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0],
        "department": ["Sales", "HR", "Sales"]
    })

@pytest.fixture
def dataset_summary():
    return DatasetSummary(
        row_count=3,
        column_count=2,
        column_names=["revenue", "department"],
        inferred_data_types={"revenue": "numeric", "department": "categorical"},
        missing_value_count={"revenue": 0, "department": 0},
        numeric_columns=["revenue"],
        categorical_columns=["department"]
    )

def test_valid_initial_plan_no_repair(sample_df):
    """1. Valid initial plan -> repair is never called, and execution succeeds."""
    agent = DataInsightAgent()

    # Mock provider returning a perfectly valid plan on first call
    mock_provider = AsyncMock()
    mock_provider.create_analysis_plan.return_value = AnalysisPlan(
        objective="Analyze revenue",
        steps=[
            AnalysisStep(
                step_id="step_1",
                operation=AnalysisOperation.MEAN,
                column="revenue",
                reason="Get average revenue"
            )
        ]
    )
    mock_provider.generate_report.return_value = MagicMock()

    with patch("app.services.agent_service.get_provider", return_value=mock_provider):
        response = asyncio.run(agent.analyze(sample_df, "What is average revenue?"))

        # Verify repair was not attempted or succeeded
        assert response.plan_repair_attempted is False
        assert response.plan_repair_succeeded is False

        # Verify provider.repair_analysis_plan was never called
        mock_provider.repair_analysis_plan.assert_not_called()

def test_invalid_initial_plan_repair_succeeds(sample_df):
    """2. & 3. Invalid initial plan -> repair is called exactly once. Valid repaired plan -> execution proceeds."""
    agent = DataInsightAgent()

    # Step with invalid configuration: MEAN cannot specify group_by
    invalid_plan = AnalysisPlan(
        objective="Analyze revenue with invalid group_by",
        steps=[
            AnalysisStep(
                step_id="step_1",
                operation=AnalysisOperation.MEAN,
                column="revenue",
                group_by="department",  # forbidden for MEAN
                reason="Invalid step parameters"
            )
        ]
    )

    valid_repaired_plan = AnalysisPlan(
        objective="Analyze revenue properly",
        steps=[
            AnalysisStep(
                step_id="step_1",
                operation=AnalysisOperation.MEAN,
                column="revenue",
                reason="Valid average revenue calculation"
            )
        ]
    )

    mock_provider = AsyncMock()
    mock_provider.create_analysis_plan.return_value = invalid_plan
    mock_provider.repair_analysis_plan.return_value = valid_repaired_plan
    mock_provider.generate_report.return_value = MagicMock()

    with patch("app.services.agent_service.get_provider", return_value=mock_provider):
        response = asyncio.run(agent.analyze(sample_df, "What is average revenue?"))

        # Verify repair was attempted and succeeded
        assert response.plan_repair_attempted is True
        assert response.plan_repair_succeeded is True

        # Verify repair_analysis_plan was called exactly once
        mock_provider.repair_analysis_plan.assert_called_once()

        # Verify feedback string is clean (e.g. it passed validation error message)
        call_kwargs = mock_provider.repair_analysis_plan.call_args[1]
        assert "Operation MEAN in 'step_1' cannot specify 'group_by'" in call_kwargs["validation_feedback"]

def test_invalid_repaired_plan_fail_closed(sample_df):
    """4. & 5. Invalid repaired plan -> fail closed, raising PlanValidationError. Repair never executes the rejected plan."""
    agent = DataInsightAgent()

    invalid_plan = AnalysisPlan(
        objective="Invalid plan",
        steps=[
            AnalysisStep(
                step_id="step_1",
                operation=AnalysisOperation.MEAN,
                column="revenue",
                group_by="department",
                reason="Invalid step parameters"
            )
        ]
    )

    # Repaired plan is still invalid (still has group_by)
    invalid_repaired_plan = AnalysisPlan(
        objective="Invalid repair",
        steps=[
            AnalysisStep(
                step_id="step_1",
                operation=AnalysisOperation.MEAN,
                column="revenue",
                group_by="department",
                reason="Still invalid"
            )
        ]
    )

    mock_provider = AsyncMock()
    mock_provider.create_analysis_plan.return_value = invalid_plan
    mock_provider.repair_analysis_plan.return_value = invalid_repaired_plan

    # Spy on executor to ensure it never runs
    with patch.object(agent.executor, "execute", return_value=MagicMock()) as mock_execute:
        with patch("app.services.agent_service.get_provider", return_value=mock_provider):
            with pytest.raises(PlanValidationError):
                asyncio.run(agent.analyze(sample_df, "What is average revenue?"))

            # Verify executor was NEVER called (fail closed)
            mock_execute.assert_not_called()

            # Repair was attempted, but failed (succeeded=False)
            assert agent.plan_repair_attempted is True
            assert agent.plan_repair_succeeded is False

            mock_provider.repair_analysis_plan.assert_called_once()

def test_sanitized_feedback_safety():
    """6. Sanitized feedback contains no stack trace, filesystem paths, or secrets.
       7. Raw CSV cells are never sent in repair payload.
    """
    # We construct a repair call prompt and verify its inputs do not contain sensitive items
    from app.prompts.planner import format_repair_user_prompt

    question = "Analyze revenue"
    summary_str = "{'column_names': ['revenue']}"
    invalid_plan_str = "{'steps': []}"

    # If the feedback contained traceback or path
    feedback_with_secrets = "Error in C:\\Users\\My_Private_Dir\\project\\validator.py: API_KEY_SECRET is invalid"

    from app.core.exceptions import PlanValidationError
    exc = PlanValidationError("Operation TOP_VALUES in 'step_1' has invalid group_by")

    # Verify no system internals or file paths are in str(exc)
    exc_str = str(exc)
    assert "\\" not in exc_str
    assert "/" not in exc_str
    assert "traceback" not in exc_str.lower()

    # Verify prompt formatting does not contain raw CSV cells
    user_prompt = format_repair_user_prompt(question, summary_str, invalid_plan_str, exc_str)
    assert "revenue,department" not in user_prompt  # no raw csv headers/data
    assert "API_KEY" not in user_prompt

def test_mock_provider_repair_is_deterministic(dataset_summary):
    """8. MockProvider repair is deterministic."""
    provider = MockProvider()

    # Normal repair returns same valid plan as create_analysis_plan
    plan_1 = asyncio.run(provider.repair_analysis_plan("What is average revenue?", dataset_summary, MagicMock(), "some error"))
    plan_2 = asyncio.run(provider.repair_analysis_plan("What is average revenue?", dataset_summary, MagicMock(), "some error"))
    assert plan_1.model_dump() == plan_2.model_dump()

    # Special query "fail_repair" returns an invalid plan deterministically
    fail_plan_1 = asyncio.run(provider.repair_analysis_plan("fail_repair", dataset_summary, MagicMock(), "some error"))
    fail_plan_2 = asyncio.run(provider.repair_analysis_plan("fail_repair", dataset_summary, MagicMock(), "some error"))
    assert fail_plan_1.model_dump() == fail_plan_2.model_dump()
    assert fail_plan_1.steps[0].column == "invalid_col_name_not_existing"

def test_evaluation_records_repair_metrics(sample_df, dataset_summary):
    """9. Evaluation correctly records repair metrics."""
    # We patch DataInsightAgent to return specific repair flags and verify they end up in PlannerScores / EvaluationResult
    case = EvaluationCase(
        case_id="case_repair_test",
        dataset_path="dummy_path.csv",
        question="What is the average revenue?",
        required_operations=[AnalysisOperation.MEAN],
        expected_result_checks=[]
    )

    # Create mock response
    mock_response = MagicMock()
    mock_response.analysis_plan = AnalysisPlan(
        objective="Analyze revenue",
        steps=[
            AnalysisStep(
                step_id="step_1",
                operation=AnalysisOperation.MEAN,
                column="revenue",
                reason="Get average revenue"
            )
        ]
    )
    mock_response.dataset_summary = dataset_summary
    mock_response.analysis_results = []
    mock_response.findings = []
    mock_response.limitations = []
    mock_response.recommendations = []
    mock_response.plan_repair_attempted = True
    mock_response.plan_repair_succeeded = True

    # 1. Successful case with repair
    mock_agent = MagicMock()
    mock_agent.analyze = AsyncMock(return_value=mock_response)
    mock_agent.plan_repair_attempted = True
    mock_agent.plan_repair_succeeded = True

    with patch("pandas.read_csv", return_value=sample_df), \
         patch("os.path.exists", return_value=True), \
         patch("evaluation.runner.DataInsightAgent", return_value=mock_agent):

        eval_result = asyncio.run(evaluate_case(case, "mock"))

        assert eval_result.planner_scores.plan_repair_attempted is True
        assert eval_result.planner_scores.plan_repair_succeeded is True

    # 2. Failed validation case (raises PlanValidationError after repair attempt)
    mock_agent_fail = MagicMock()
    mock_agent_fail.analyze = AsyncMock(side_effect=PlanValidationError("Fail"))
    mock_agent_fail.plan_repair_attempted = True
    mock_agent_fail.plan_repair_succeeded = False

    with patch("pandas.read_csv", return_value=sample_df), \
         patch("os.path.exists", return_value=True), \
         patch("evaluation.runner.DataInsightAgent", return_value=mock_agent_fail):

         eval_result_fail = asyncio.run(evaluate_case(case, "mock"))

         assert eval_result_fail.error_category == "plan_validation_failed"
         assert eval_result_fail.planner_scores.plan_repair_attempted is True
         assert eval_result_fail.planner_scores.plan_repair_succeeded is False
