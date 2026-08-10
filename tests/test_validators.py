import pytest
from app.models.responses import DatasetSummary
from app.models.analysis import (
    AnalysisPlan,
    AnalysisStep,
    AnalysisOperation,
    ProviderReport,
    Finding,
    Recommendation,
    RecommendationPriority,
    ConfidenceLevel,
    AnalysisResult
)
from app.services.plan_validator import PlanValidator
from app.services.grounding_validator import GroundingValidator
from app.core.exceptions import PlanValidationError, GroundingValidationError

@pytest.fixture
def sample_summary():
    return DatasetSummary(
        row_count=10,
        column_count=3,
        column_names=["revenue", "orders", "department"],
        inferred_data_types={
            "revenue": "numeric",
            "orders": "numeric",
            "department": "categorical"
        },
        missing_value_count={"revenue": 0, "orders": 0, "department": 0},
        numeric_columns=["revenue", "orders"],
        categorical_columns=["department"]
    )

# Plan safety validator tests

def test_plan_validator_success(sample_summary):
    plan = AnalysisPlan(
        objective="Calculate average sales",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.COUNT),
            AnalysisStep(step_id="step_2", operation=AnalysisOperation.MEAN, column="revenue"),
            AnalysisStep(step_id="step_3", operation=AnalysisOperation.TOP_VALUES, column="department", limit=10),
            AnalysisStep(step_id="step_4", operation=AnalysisOperation.GROUP_BY_MEAN, column="revenue", group_by="department"),
            AnalysisStep(step_id="step_5", operation=AnalysisOperation.CORRELATION, column="revenue", second_column="orders")
        ]
    )
    # Should not raise exception
    PlanValidator.validate(plan, sample_summary)

def test_plan_validator_empty():
    plan = AnalysisPlan(objective="do nothing", steps=[])
    summary = DatasetSummary(
        row_count=0, column_count=0, column_names=[], inferred_data_types={},
        missing_value_count={}, numeric_columns=[], categorical_columns=[]
    )
    with pytest.raises(PlanValidationError) as exc:
        PlanValidator.validate(plan, summary)
    assert "cannot be empty" in str(exc.value)

def test_plan_validator_too_many_steps(sample_summary):
    # settings.MAX_ANALYSIS_STEPS defaults to 8
    steps = [
        AnalysisStep(step_id=f"step_{i}", operation=AnalysisOperation.COUNT) for i in range(1, 10)
    ]
    plan = AnalysisPlan(objective="too many steps", steps=steps)
    with pytest.raises(PlanValidationError) as exc:
        PlanValidator.validate(plan, sample_summary)
    assert "exceeds configured maximum limit" in str(exc.value)

def test_plan_validator_duplicate_step_ids(sample_summary):
    plan = AnalysisPlan(
        objective="dup ids",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.COUNT),
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEAN, column="revenue")
        ]
    )
    with pytest.raises(PlanValidationError) as exc:
        PlanValidator.validate(plan, sample_summary)
    assert "Duplicate step ID" in str(exc.value)

def test_plan_validator_duplicate_equivalent_steps(sample_summary):
    plan = AnalysisPlan(
        objective="dup equivalent steps",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEAN, column="revenue"),
            AnalysisStep(step_id="step_2", operation=AnalysisOperation.MEAN, column="revenue")
        ]
    )
    with pytest.raises(PlanValidationError) as exc:
        PlanValidator.validate(plan, sample_summary)
    assert "Duplicate equivalent step detected" in str(exc.value)

def test_plan_validator_nonexistent_column(sample_summary):
    plan = AnalysisPlan(
        objective="bad col",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEAN, column="nonexistent")
        ]
    )
    with pytest.raises(PlanValidationError) as exc:
        PlanValidator.validate(plan, sample_summary)
    assert "does not exist in dataset" in str(exc.value)

def test_plan_validator_numeric_operation_on_categorical(sample_summary):
    plan = AnalysisPlan(
        objective="bad type",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEAN, column="department")
        ]
    )
    with pytest.raises(PlanValidationError) as exc:
        PlanValidator.validate(plan, sample_summary)
    assert "requires numeric column" in str(exc.value)

def test_plan_validator_malformed_correlation(sample_summary):
    # CORRELATION requires second_column
    plan = AnalysisPlan(
        objective="bad correlation",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.CORRELATION, column="revenue")
        ]
    )
    with pytest.raises(PlanValidationError) as exc:
        PlanValidator.validate(plan, sample_summary)
    assert "requires both" in str(exc.value)

def test_plan_validator_malformed_group_by(sample_summary):
    # GROUP_BY_MEAN requires group_by parameter
    plan = AnalysisPlan(
        objective="bad group by",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.GROUP_BY_MEAN, column="revenue")
        ]
    )
    with pytest.raises(PlanValidationError) as exc:
        PlanValidator.validate(plan, sample_summary)
    assert "requires both" in str(exc.value)

def test_plan_validator_invalid_top_values_limit(sample_summary):
    plan = AnalysisPlan(
        objective="bad limit",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.TOP_VALUES, column="department", limit=150)
        ]
    )
    with pytest.raises(PlanValidationError) as exc:
        PlanValidator.validate(plan, sample_summary)
    assert "invalid 'limit'" in str(exc.value)

def test_plan_validator_unsupported_combinations(sample_summary):
    # MEAN with extra second_column
    plan = AnalysisPlan(
        objective="bad mean",
        steps=[
            AnalysisStep(
                step_id="step_1",
                operation=AnalysisOperation.MEAN,
                column="revenue",
                second_column="orders"
            )
        ]
    )
    with pytest.raises(PlanValidationError) as exc:
        PlanValidator.validate(plan, sample_summary)
    assert "cannot specify" in str(exc.value)

# Grounding validator tests

@pytest.fixture
def sample_results():
    return [
        AnalysisResult(
            result_id="result_1", source_step_id="step_1", operation=AnalysisOperation.COUNT,
            target_columns=[], computed_result=10, description="Count total rows"
        ),
        AnalysisResult(
            result_id="result_2", source_step_id="step_2", operation=AnalysisOperation.MEAN,
            target_columns=["revenue"], computed_result=100.0, description="Mean revenue"
        )
    ]

def test_grounding_validator_success(sample_results):
    report = ProviderReport(
        findings=[
            Finding(id="finding_1", title="Scale", explanation="Count is 10.", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH),
            Finding(id="finding_2", title="Mean", explanation="Mean is 100.", evidence_refs=["result_2"], confidence=ConfidenceLevel.HIGH)
        ],
        limitations=["limited data"],
        recommendations=[
            Recommendation(id="recommendation_1", priority=RecommendationPriority.HIGH, action="Do X", rationale="Because Y", finding_refs=["finding_1", "finding_2"])
        ]
    )
    # Should not raise exception
    GroundingValidator.validate(report, sample_results)

def test_grounding_validator_unknown_evidence_id(sample_results):
    report = ProviderReport(
        findings=[
            Finding(id="finding_1", title="Scale", explanation="Count is 10.", evidence_refs=["result_99"], confidence=ConfidenceLevel.HIGH)
        ],
        limitations=["none"],
        recommendations=[]
    )
    with pytest.raises(GroundingValidationError) as exc:
        GroundingValidator.validate(report, sample_results)
    assert "references a non-existent or foreign result ID" in str(exc.value)

def test_grounding_validator_unknown_finding_id(sample_results):
    report = ProviderReport(
        findings=[
            Finding(id="finding_1", title="Scale", explanation="Count is 10.", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH)
        ],
        limitations=["none"],
        recommendations=[
            Recommendation(id="recommendation_1", priority=RecommendationPriority.MEDIUM, action="Action", rationale="Reason", finding_refs=["finding_99"])
        ]
    )
    with pytest.raises(GroundingValidationError) as exc:
        GroundingValidator.validate(report, sample_results)
    assert "references a non-existent or foreign finding ID" in str(exc.value)

def test_grounding_validator_duplicate_finding_ids(sample_results):
    report = ProviderReport(
        findings=[
            Finding(id="finding_1", title="Scale", explanation="Count is 10.", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH),
            Finding(id="finding_1", title="Scale 2", explanation="Count is 10.", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH)
        ],
        limitations=["none"],
        recommendations=[]
    )
    with pytest.raises(GroundingValidationError) as exc:
        GroundingValidator.validate(report, sample_results)
    assert "Duplicate finding ID" in str(exc.value)

def test_grounding_validator_duplicate_recommendation_ids(sample_results):
    report = ProviderReport(
        findings=[
            Finding(id="finding_1", title="Scale", explanation="Count is 10.", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH)
        ],
        limitations=["none"],
        recommendations=[
            Recommendation(id="recommendation_1", priority=RecommendationPriority.LOW, action="A", rationale="R", finding_refs=["finding_1"]),
            Recommendation(id="recommendation_1", priority=RecommendationPriority.LOW, action="B", rationale="R", finding_refs=["finding_1"])
        ]
    )
    with pytest.raises(GroundingValidationError) as exc:
        GroundingValidator.validate(report, sample_results)
    assert "Duplicate recommendation ID" in str(exc.value)

def test_grounding_validator_orphan_finding(sample_results):
    # Finding must contain at least one evidence_ref
    report = ProviderReport(
        findings=[
            Finding(id="finding_1", title="Scale", explanation="Count is 10.", evidence_refs=[], confidence=ConfidenceLevel.HIGH)
        ],
        limitations=["none"],
        recommendations=[]
    )
    with pytest.raises(GroundingValidationError) as exc:
        GroundingValidator.validate(report, sample_results)
    assert "contains no evidence references" in str(exc.value)

def test_grounding_validator_orphan_recommendation(sample_results):
    # Recommendation must contain at least one finding_ref
    report = ProviderReport(
        findings=[
            Finding(id="finding_1", title="Scale", explanation="Count is 10.", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH)
        ],
        limitations=["none"],
        recommendations=[
            Recommendation(id="recommendation_1", priority=RecommendationPriority.LOW, action="A", rationale="R", finding_refs=[])
        ]
    )
    with pytest.raises(GroundingValidationError) as exc:
        GroundingValidator.validate(report, sample_results)
    assert "contains no finding references" in str(exc.value)

def test_grounding_validator_missing_limitations(sample_results):
    report = ProviderReport(
        findings=[
            Finding(id="finding_1", title="Scale", explanation="Count is 10.", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH)
        ],
        limitations=[],
        recommendations=[]
    )
    with pytest.raises(GroundingValidationError) as exc:
        GroundingValidator.validate(report, sample_results)
    assert "must contain at least one limitation statement" in str(exc.value)
