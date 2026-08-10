import os
import math
import pandas as pd
import pytest
from app.services.analysis_executor import AnalysisExecutor
from app.models.analysis import AnalysisStep, AnalysisOperation
from app.core.exceptions import AnalysisExecutionError

@pytest.fixture
def sample_sales_df():
    """Load the deterministic sample sales CSV fixture."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "fixtures", "sample_sales.csv")
    return pd.read_csv(csv_path)

def test_count_no_column(sample_sales_df):
    executor = AnalysisExecutor()
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.COUNT)
    result = executor.execute(sample_sales_df, step)

    assert result.operation == AnalysisOperation.COUNT
    assert result.target_columns == []
    assert result.computed_result == 4
    assert "total rows" in result.description

def test_count_with_column(sample_sales_df):
    executor = AnalysisExecutor()
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.COUNT, column="revenue")
    result = executor.execute(sample_sales_df, step)

    assert result.operation == AnalysisOperation.COUNT
    assert result.target_columns == ["revenue"]
    assert result.computed_result == 4

    # Introduce a null in a copy to verify count is only non-null values
    df_copy = sample_sales_df.copy()
    df_copy.loc[0, "revenue"] = None
    result_null = executor.execute(df_copy, step)
    assert result_null.computed_result == 3

def test_mean(sample_sales_df):
    executor = AnalysisExecutor()
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEAN, column="revenue")
    result = executor.execute(sample_sales_df, step)

    assert result.operation == AnalysisOperation.MEAN
    assert result.computed_result == 300.0

def test_median(sample_sales_df):
    executor = AnalysisExecutor()
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEDIAN, column="revenue")
    result = executor.execute(sample_sales_df, step)

    assert result.operation == AnalysisOperation.MEDIAN
    assert result.computed_result == 300.0

def test_sum(sample_sales_df):
    executor = AnalysisExecutor()
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.SUM, column="revenue")
    result = executor.execute(sample_sales_df, step)

    assert result.operation == AnalysisOperation.SUM
    assert result.computed_result == 1200.0

def test_min_max(sample_sales_df):
    executor = AnalysisExecutor()
    step_min = AnalysisStep(step_id="step_1", operation=AnalysisOperation.MIN, column="revenue")
    result_min = executor.execute(sample_sales_df, step_min)
    assert result_min.computed_result == 100.0

    step_max = AnalysisStep(step_id="step_2", operation=AnalysisOperation.MAX, column="revenue")
    result_max = executor.execute(sample_sales_df, step_max)
    assert result_max.computed_result == 500.0

def test_std(sample_sales_df):
    executor = AnalysisExecutor()
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.STD, column="revenue")
    result = executor.execute(sample_sales_df, step)

    assert result.operation == AnalysisOperation.STD
    # std of [100, 200, 400, 500] ddof=1 is approx 182.574
    assert pytest.approx(result.computed_result, rel=1e-4) == 182.57418

def test_missing_values(sample_sales_df):
    executor = AnalysisExecutor()
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.MISSING_VALUES, column="revenue")
    result = executor.execute(sample_sales_df, step)
    assert result.computed_result == 0

    # Introduce nulls
    df_copy = sample_sales_df.copy()
    df_copy.loc[0, "revenue"] = None
    result_null = executor.execute(df_copy, step)
    assert result_null.computed_result == 1

def test_unique_count(sample_sales_df):
    executor = AnalysisExecutor()
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.UNIQUE_COUNT, column="department")
    result = executor.execute(sample_sales_df, step)
    assert result.computed_result == 2

def test_top_values(sample_sales_df):
    executor = AnalysisExecutor()
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.TOP_VALUES, column="department", limit=5)
    result = executor.execute(sample_sales_df, step)

    assert result.operation == AnalysisOperation.TOP_VALUES
    assert result.computed_result == {"A": 2, "B": 2}

def test_group_by_mean(sample_sales_df):
    executor = AnalysisExecutor()
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.GROUP_BY_MEAN, column="revenue", group_by="department")
    result = executor.execute(sample_sales_df, step)

    assert result.operation == AnalysisOperation.GROUP_BY_MEAN
    assert result.grouping_column == "department"
    assert result.computed_result == {"A": 150.0, "B": 450.0}

def test_group_by_count(sample_sales_df):
    executor = AnalysisExecutor()
    # GROUP_BY_COUNT without target column
    step_no_target = AnalysisStep(step_id="step_1", operation=AnalysisOperation.GROUP_BY_COUNT, group_by="department")
    result_no_target = executor.execute(sample_sales_df, step_no_target)
    assert result_no_target.computed_result == {"A": 2, "B": 2}

    # GROUP_BY_COUNT with target column
    step_target = AnalysisStep(step_id="step_2", operation=AnalysisOperation.GROUP_BY_COUNT, group_by="department", column="revenue")
    result_target = executor.execute(sample_sales_df, step_target)
    assert result_target.computed_result == {"A": 2, "B": 2}

def test_correlation(sample_sales_df):
    executor = AnalysisExecutor()
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.CORRELATION, column="revenue", second_column="orders")
    result = executor.execute(sample_sales_df, step)

    assert result.operation == AnalysisOperation.CORRELATION
    assert pytest.approx(result.computed_result, rel=1e-4) == 1.0

# Negative Executor Tests

def test_negative_missing_parameters(sample_sales_df):
    executor = AnalysisExecutor()

    # MEAN missing target column
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEAN)
    with pytest.raises(AnalysisExecutionError) as exc:
        executor.execute(sample_sales_df, step)
    assert "requires a target column" in str(exc.value)

def test_negative_invalid_limit(sample_sales_df):
    executor = AnalysisExecutor()

    # TOP_VALUES with zero limit
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.TOP_VALUES, column="department", limit=0)
    with pytest.raises(AnalysisExecutionError) as exc:
        executor.execute(sample_sales_df, step)
    assert "requires a limit between 1 and 100" in str(exc.value)

    # TOP_VALUES with limit > 100
    step = AnalysisStep(step_id="step_2", operation=AnalysisOperation.TOP_VALUES, column="department", limit=101)
    with pytest.raises(AnalysisExecutionError) as exc:
        executor.execute(sample_sales_df, step)
    assert "requires a limit between 1 and 100" in str(exc.value)

def test_negative_invalid_combinations(sample_sales_df):
    executor = AnalysisExecutor()

    # GROUP_BY_MEAN missing group_by
    step_missing_group_by = AnalysisStep(step_id="step_1", operation=AnalysisOperation.GROUP_BY_MEAN, column="revenue")
    with pytest.raises(AnalysisExecutionError) as exc:
        executor.execute(sample_sales_df, step_missing_group_by)
    assert "requires a grouping column" in str(exc.value)

    # CORRELATION missing second_column
    step_missing_second = AnalysisStep(step_id="step_2", operation=AnalysisOperation.CORRELATION, column="revenue")
    with pytest.raises(AnalysisExecutionError) as exc:
        executor.execute(sample_sales_df, step_missing_second)
    assert "requires both" in str(exc.value)

def test_negative_nonexistent_column(sample_sales_df):
    executor = AnalysisExecutor()

    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEAN, column="nonexistent_column")
    with pytest.raises(AnalysisExecutionError) as exc:
        executor.execute(sample_sales_df, step)
    assert "does not exist in dataset" in str(exc.value)

def test_negative_unsupported_dtype(sample_sales_df):
    executor = AnalysisExecutor()

    # Running MEAN on text column 'department'
    step = AnalysisStep(step_id="step_1", operation=AnalysisOperation.MEAN, column="department")
    with pytest.raises(AnalysisExecutionError) as exc:
        executor.execute(sample_sales_df, step)
    assert "must be numeric" in str(exc.value)
