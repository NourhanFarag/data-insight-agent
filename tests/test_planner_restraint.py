import pytest
import pandas as pd
import asyncio
from unittest.mock import patch
from app.models.responses import DatasetSummary
from app.models.analysis import (
    AnalysisPlan,
    AnalysisStep,
    AnalysisResult,
    AnalysisOperation,
    ProviderReport,
    Finding,
    Recommendation,
    ConfidenceLevel,
    RecommendationPriority
)
from app.services.agent_service import DataInsightAgent
from app.prompts.planner import PLANNER_SYSTEM_PROMPT, REPAIR_SYSTEM_PROMPT
from app.prompts.reporter import REPORTER_SYSTEM_PROMPT
from app.providers.mock_provider import MockProvider
from app.core.exceptions import PlanValidationError


# 1. Prompt-Contract Regression Tests
def test_prompt_contract_planner_restraints():
    """Verify that PLANNER_SYSTEM_PROMPT and REPAIR_SYSTEM_PROMPT contain the general unsupported-question restraint instructions."""
    # Assert specific keywords/rules are present in the prompts
    for prompt_name, prompt in [("PLANNER_SYSTEM_PROMPT", PLANNER_SYSTEM_PROMPT), ("REPAIR_SYSTEM_PROMPT", REPAIR_SYSTEM_PROMPT)]:
        assert "UNSUPPORTED-QUESTION RESTRAINT RULE" in prompt
        assert "materially support" in prompt or "DatasetSummary" in prompt
        assert "Do NOT invent proxy metrics" in prompt
        assert "Do NOT perform exploratory GROUP_BY_MEAN" in prompt or "unrelated fields" in prompt
        assert "COUNT" in prompt
        assert "Column: Omit" in prompt or "Column: leave empty" in prompt or "reason" in prompt


def test_prompt_contract_reporter_restraints():
    """Verify that REPORTER_SYSTEM_PROMPT contains the general unsupported-question fallback interpretation instructions."""
    assert "UNSUPPORTED-QUESTION fallbacks" in REPORTER_SYSTEM_PROMPT
    assert "not represented" in REPORTER_SYSTEM_PROMPT
    assert "COUNT" in REPORTER_SYSTEM_PROMPT
    assert "Do NOT treat or present the fallback COUNT result as evidence" in REPORTER_SYSTEM_PROMPT
    assert "additional data" in REPORTER_SYSTEM_PROMPT


# Stub Provider for Pipeline Orchestration Verification
class StubProvider(MockProvider):
    def __init__(self, plan_to_return, report_to_return):
        self.plan_to_return = plan_to_return
        self.report_to_return = report_to_return
        self.create_plan_calls = []
        self.generate_report_calls = []
        self.repair_calls = []

    async def create_analysis_plan(self, question: str, summary: DatasetSummary) -> AnalysisPlan:
        self.create_plan_calls.append((question, summary))
        return self.plan_to_return

    async def repair_analysis_plan(
        self,
        question: str,
        dataset_summary: DatasetSummary,
        invalid_plan: AnalysisPlan,
        validation_feedback: str,
    ) -> AnalysisPlan:
        self.repair_calls.append((question, dataset_summary, invalid_plan, validation_feedback))
        return self.plan_to_return

    async def generate_report(
        self, question: str, summary: DatasetSummary, results: list[AnalysisResult]
    ) -> ProviderReport:
        self.generate_report_calls.append((question, summary, results))
        return self.report_to_return


# 2. Orchestration / Reporting Tests
def test_unsupported_concept_fallback_pipeline():
    """Verify that the orchestration pipeline handles the fallback COUNT plan correctly for unsupported questions."""
    df = pd.DataFrame({
        "department": ["Sales", "Engineering"],
        "revenue": [100.0, 200.0]
    })
    question = "Why did customers become dissatisfied with support agents?"

    # Stub plan returning the minimal fallback COUNT
    fallback_plan = AnalysisPlan(
        objective="Fallback row count since requested concept is unsupported",
        steps=[
            AnalysisStep(
                step_id="step_1",
                operation=AnalysisOperation.COUNT,
                reason="The available columns do not contain support dissatisfaction data."
            )
        ]
    )

    # Stub report that states data cannot answer the question and identifies missing data conceptually
    stub_report = ProviderReport(
        findings=[
            Finding(
                id="finding_1",
                title="Data Limitation",
                explanation="The available dataset does not contain support agent feedback or dissatisfaction metrics. A count of 2 rows was performed for context only.",
                evidence_refs=["result_1"],
                confidence=ConfidenceLevel.LOW
            )
        ],
        limitations=[
            "Causal analysis of customer dissatisfaction is not possible with the provided fields."
        ],
        recommendations=[
            Recommendation(
                id="recommendation_1",
                priority=RecommendationPriority.LOW,
                action="Collect support ticket CSAT scores and satisfaction surveys.",
                rationale="We need support-related columns to perform satisfaction analysis.",
                finding_refs=["finding_1"]
            )
        ]
    )

    stub_provider = StubProvider(fallback_plan, stub_report)

    with patch("app.services.agent_service.get_provider", return_value=stub_provider):
        agent = DataInsightAgent()
        response = asyncio.run(agent.analyze(df, question))

        # 1. Assert plan fallback was executed
        assert len(response.analysis_plan.steps) == 1
        assert response.analysis_plan.steps[0].operation == AnalysisOperation.COUNT
        assert response.analysis_plan.steps[0].column is None

        # 2. Assert no unrelated GROUP_BY_MEAN or CORRELATION proxy operations are in the plan
        for step in response.analysis_plan.steps:
            assert step.operation != AnalysisOperation.GROUP_BY_MEAN
            assert step.operation != AnalysisOperation.CORRELATION

        # 7. Assert reporter explicitly states the limitation
        assert "does not contain support agent feedback" in response.findings[0].explanation
        assert "not possible" in response.limitations[0]

        # 8. Assert reporter does not treat COUNT as evidence for the requested cause
        # (It shouldn't say the count of rows explains satisfaction)
        assert "dissatisfied" not in response.findings[0].explanation.lower() or "does not contain" in response.findings[0].explanation


def test_supported_questions_plan_normally():
    """Verify that supported questions still plan and execute normally."""
    df = pd.DataFrame({
        "revenue": [100.0, 200.0],
        "department": ["Sales", "Engineering"]
    })

    # Test Case 4: Supported revenue question plans normally
    plan_revenue = AnalysisPlan(
        objective="Calculate average revenue",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEAN, column="revenue", reason="Compute average revenue")
        ]
    )
    report_revenue = ProviderReport(
        findings=[Finding(id="finding_1", title="Revenue", explanation="Average revenue is 150", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH)],
        limitations=["The dataset is limited to basic financial records."], recommendations=[]
    )
    
    with patch("app.services.agent_service.get_provider", return_value=StubProvider(plan_revenue, report_revenue)):
        agent = DataInsightAgent()
        resp = asyncio.run(agent.analyze(df, "What is the average revenue?"))
        assert resp.analysis_plan.steps[0].operation == AnalysisOperation.MEAN
        assert resp.analysis_plan.steps[0].column == "revenue"

    # Test Case 5: Supported grouped question plans normally
    plan_grouped = AnalysisPlan(
        objective="Calculate revenue by department",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.GROUP_BY_MEAN, column="revenue", group_by="department", reason="Revenue by dept")
        ]
    )
    report_grouped = ProviderReport(
        findings=[Finding(id="finding_1", title="Grouped Revenue", explanation="Revenue by dept shows variation", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH)],
        limitations=["The dataset is limited to basic financial records."], recommendations=[]
    )
    with patch("app.services.agent_service.get_provider", return_value=StubProvider(plan_grouped, report_grouped)):
        agent = DataInsightAgent()
        resp = asyncio.run(agent.analyze(df, "Average revenue by department?"))
        assert resp.analysis_plan.steps[0].operation == AnalysisOperation.GROUP_BY_MEAN
        assert resp.analysis_plan.steps[0].group_by == "department"

    # Test Case 6: Supported correlation question plans normally
    df_corr = pd.DataFrame({
        "revenue": [100.0, 200.0],
        "orders": [5, 10]
    })
    plan_corr = AnalysisPlan(
        objective="Correlation between revenue and orders",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.CORRELATION, column="revenue", second_column="orders", reason="Corr")
        ]
    )
    report_corr = ProviderReport(
        findings=[Finding(id="finding_1", title="Correlation", explanation="Corr is 1.0", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH)],
        limitations=["The dataset is limited to basic financial records."], recommendations=[]
    )
    with patch("app.services.agent_service.get_provider", return_value=StubProvider(plan_corr, report_corr)):
        agent = DataInsightAgent()
        resp = asyncio.run(agent.analyze(df_corr, "Correlation between revenue and orders?"))
        assert resp.analysis_plan.steps[0].operation == AnalysisOperation.CORRELATION
        assert resp.analysis_plan.steps[0].second_column == "orders"


def test_plan_repair_behavior_unchanged():
    """Verify that when a plan is invalid, the repair agent is called."""
    df = pd.DataFrame({
        "revenue": [100.0, 200.0]
    })

    # Return an invalid plan first (uses a non-existing column)
    invalid_plan = AnalysisPlan(
        objective="Exploratory",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEAN, column="non_existent_column", reason="Mean")
        ]
    )

    # Valid plan to return after repair
    repaired_plan = AnalysisPlan(
        objective="Exploratory",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEAN, column="revenue", reason="Mean")
        ]
    )

    report = ProviderReport(
        findings=[Finding(id="finding_1", title="Revenue", explanation="Average revenue is 150", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH)],
        limitations=["The dataset is limited to basic financial records."], recommendations=[]
    )

    # We stub a provider that returns the invalid plan first, then repaired plan on repair
    class RepairStubProvider(StubProvider):
        async def create_analysis_plan(self, question: str, summary: DatasetSummary) -> AnalysisPlan:
            return invalid_plan

        async def repair_analysis_plan(
            self, question: str, dataset_summary: DatasetSummary, invalid_plan: AnalysisPlan, validation_feedback: str
        ) -> AnalysisPlan:
            self.repair_calls.append((question, dataset_summary, invalid_plan, validation_feedback))
            return repaired_plan

    stub_provider = RepairStubProvider(repaired_plan, report)

    with patch("app.services.agent_service.get_provider", return_value=stub_provider):
        agent = DataInsightAgent()
        resp = asyncio.run(agent.analyze(df, "What is the average revenue?"))
        
        # Verify repair was attempted and succeeded
        assert resp.plan_repair_attempted is True
        assert resp.plan_repair_succeeded is True
        assert len(stub_provider.repair_calls) == 1
        assert resp.analysis_plan.steps[0].column == "revenue"


# 3. Benchmark Case and Files Integrity Test
def test_benchmark_evaluation_files_unchanged():
    """Verify that no benchmark evaluation cases or histories have been modified post-tuning."""
    from evaluation.runner import load_cases
    cases = load_cases("evaluation/cases")
    assert len(cases) == 10

    case_map = {c.case_id: c for c in cases}
    assert case_map["category_frequency"].question == "What is the frequency count for every customer segment?"
    assert case_map["missing_data"].question == "How many missing values are in each field, including fields with zero missing values?"
    assert case_map["adversarial_case"].question == "Show the frequency count for every distinct value in the department column."
    assert case_map["unsupported_question"].question == "Why did customers become dissatisfied with support agents?"


def test_provider_error_handling_agent_service():
    """Verify that when generate_report() raises a ProviderError:
       - No NameError occurs (ProviderError is properly imported)
       - analysis_plan is attached
       - analysis_results are attached
       - the original ProviderError is re-raised
    """
    from app.core.exceptions import ProviderError

    df = pd.DataFrame({
        "revenue": [100.0, 200.0]
    })

    plan = AnalysisPlan(
        objective="Calculate average revenue",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEAN, column="revenue", reason="Compute average revenue")
        ]
    )

    class ErrorStubProvider(StubProvider):
        async def generate_report(self, question, summary, results):
            raise ProviderError("Mocked LLM generation failure", status_code=502)

    stub_provider = ErrorStubProvider(plan, None)

    with patch("app.services.agent_service.get_provider", return_value=stub_provider):
        agent = DataInsightAgent()
        with pytest.raises(ProviderError) as exc_info:
            asyncio.run(agent.analyze(df, "What is the average revenue?"))

        err = exc_info.value
        assert "Mocked LLM generation failure" in str(err)
        assert err.analysis_plan == plan
        assert len(err.analysis_results) == 1
        assert err.analysis_results[0].operation == AnalysisOperation.MEAN
