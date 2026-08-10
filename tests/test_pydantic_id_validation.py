import pytest
from pydantic import ValidationError
from app.models.analysis import AnalysisStep, AnalysisResult, AnalysisOperation

def test_pydantic_validation_missing_step_id():
    with pytest.raises(ValidationError):
        # Missing step_id entirely
        AnalysisStep(operation=AnalysisOperation.COUNT)

def test_pydantic_validation_blank_step_id():
    with pytest.raises(ValidationError):
        # Blank step_id
        AnalysisStep(step_id="", operation=AnalysisOperation.COUNT)
    with pytest.raises(ValidationError):
        # Whitespace step_id (does not match pattern)
        AnalysisStep(step_id="   ", operation=AnalysisOperation.COUNT)

def test_pydantic_validation_malformed_step_id():
    with pytest.raises(ValidationError):
        # Must match pattern step_[1-9]\d* (no step_0 allowed)
        AnalysisStep(step_id="step_0", operation=AnalysisOperation.COUNT)
    with pytest.raises(ValidationError):
        # Leading zero not allowed
        AnalysisStep(step_id="step_05", operation=AnalysisOperation.COUNT)
    with pytest.raises(ValidationError):
        # Alpha suffix not allowed
        AnalysisStep(step_id="step_1a", operation=AnalysisOperation.COUNT)
    with pytest.raises(ValidationError):
        # Wrong prefix
        AnalysisStep(step_id="other_1", operation=AnalysisOperation.COUNT)

def test_pydantic_validation_missing_result_id():
    with pytest.raises(ValidationError):
        # Missing result_id entirely
        AnalysisResult(
            source_step_id="step_1",
            operation=AnalysisOperation.COUNT,
            target_columns=[],
            computed_result=10,
            description="Count"
        )

def test_pydantic_validation_blank_result_id():
    with pytest.raises(ValidationError):
        # Blank result_id
        AnalysisResult(
            result_id="",
            source_step_id="step_1",
            operation=AnalysisOperation.COUNT,
            target_columns=[],
            computed_result=10,
            description="Count"
        )

def test_pydantic_validation_malformed_result_id():
    with pytest.raises(ValidationError):
        # Must match result_[1-9]\d*
        AnalysisResult(
            result_id="result_0",
            source_step_id="step_1",
            operation=AnalysisOperation.COUNT,
            target_columns=[],
            computed_result=10,
            description="Count"
        )
    with pytest.raises(ValidationError):
        # Leading zero not allowed
        AnalysisResult(
            result_id="result_01",
            source_step_id="step_1",
            operation=AnalysisOperation.COUNT,
            target_columns=[],
            computed_result=10,
            description="Count"
        )

def test_pydantic_validation_missing_source_step_id():
    with pytest.raises(ValidationError):
        # Missing source_step_id entirely
        AnalysisResult(
            result_id="result_1",
            operation=AnalysisOperation.COUNT,
            target_columns=[],
            computed_result=10,
            description="Count"
        )

def test_pydantic_validation_malformed_source_step_id():
    with pytest.raises(ValidationError):
        # source_step_id must match step_[1-9]\d*
        AnalysisResult(
            result_id="result_1",
            source_step_id="step_0",
            operation=AnalysisOperation.COUNT,
            target_columns=[],
            computed_result=10,
            description="Count"
        )
