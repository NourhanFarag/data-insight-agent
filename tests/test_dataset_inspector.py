import pandas as pd
from app.services.dataset_inspector import DatasetInspector

def test_dataset_inspector_success():
    # Build a DataFrame with numeric columns, text columns, and nulls
    df = pd.DataFrame({
        "revenue": [100.0, 200.0, None, 400.0],
        "orders": [1, 2, 3, 4],
        "department": ["A", "B", "A", "B"]
    })

    summary = DatasetInspector.inspect(df)

    # Assert row/col counts
    assert summary.row_count == 4
    assert summary.column_count == 3
    assert summary.column_names == ["revenue", "orders", "department"]

    # Assert type classification
    assert "revenue" in summary.numeric_columns
    assert "orders" in summary.numeric_columns
    assert "department" in summary.categorical_columns

    # Assert missing values
    assert summary.missing_value_count["revenue"] == 1
    assert summary.missing_value_count["orders"] == 0
    assert summary.missing_value_count["department"] == 0

    # Assert type mapping
    assert summary.inferred_data_types["revenue"] == "numeric"
    assert summary.inferred_data_types["orders"] == "numeric"
    assert summary.inferred_data_types["department"] == "categorical"
