import asyncio
from app.providers.mock_provider import MockProvider
from app.models.responses import DatasetSummary
from app.models.analysis import AnalysisOperation, ConfidenceLevel, AnalysisResult

def test_mock_provider_plan_and_report():
    provider = MockProvider()

    # Construct a DatasetSummary mock input
    summary = DatasetSummary(
        row_count=100,
        column_count=3,
        column_names=["revenue", "orders", "department"],
        inferred_data_types={
            "revenue": "numeric",
            "orders": "numeric",
            "department": "categorical"
        },
        missing_value_count={
            "revenue": 0,
            "orders": 0,
            "department": 0
        },
        numeric_columns=["revenue", "orders"],
        categorical_columns=["department"]
    )

    # 1. Test create_analysis_plan
    plan = asyncio.run(provider.create_analysis_plan("What is the average revenue?", summary))

    # Verify deterministic plan
    assert len(plan.steps) == 4
    assert plan.steps[0].operation == AnalysisOperation.COUNT
    assert plan.steps[1].operation == AnalysisOperation.MEAN
    assert plan.steps[1].column == "revenue"
    assert plan.steps[2].operation == AnalysisOperation.GROUP_BY_MEAN
    assert plan.steps[2].column == "revenue"
    assert plan.steps[2].group_by == "department"
    assert plan.steps[3].operation == AnalysisOperation.TOP_VALUES
    assert plan.steps[3].column == "department"

    # 2. Test generate_report
    results = [
        AnalysisResult(
            result_id="result_1",
            source_step_id="step_1",
            operation=AnalysisOperation.COUNT,
            target_columns=[],
            computed_result=100,
            description="Counted total rows in dataset. Result: 100."
        ),
        AnalysisResult(
            result_id="result_2",
            source_step_id="step_2",
            operation=AnalysisOperation.MEAN,
            target_columns=["revenue"],
            computed_result=250.0,
            description="Computed mean on column 'revenue'. Result: 250.0."
        )
    ]

    report = asyncio.run(provider.generate_report("What is the average revenue?", summary, results))

    # Verify deterministic report fields
    assert len(report.findings) == 2
    assert report.findings[0].id == "finding_1"
    assert "100" in report.findings[0].explanation
    assert report.findings[0].evidence_refs == ["result_1"]

    assert report.findings[1].id == "finding_2"
    assert "250.0" in report.findings[1].explanation
    assert report.findings[1].evidence_refs == ["result_2"]

    assert len(report.recommendations) > 0
    assert report.recommendations[0].finding_refs == ["finding_1"]
    assert len(report.limitations) > 0
