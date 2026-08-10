import pandas as pd
from app.models.responses import DatasetSummary

class DatasetInspector:
    @staticmethod
    def inspect(df: pd.DataFrame) -> DatasetSummary:
        """Inspects a pandas DataFrame and returns a structured DatasetSummary."""
        row_count = len(df)
        column_names = [str(col) for col in df.columns]
        column_count = len(column_names)

        inferred_data_types = {}
        numeric_columns = []
        categorical_columns = []

        # Calculate missing values
        missing_values_raw = df.isnull().sum().to_dict()
        missing_value_count = {str(col): int(count) for col, count in missing_values_raw.items()}

        for col in df.columns:
            col_str = str(col)
            dtype = df[col].dtype

            if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                inferred_data_types[col_str] = "numeric"
                numeric_columns.append(col_str)
            elif pd.api.types.is_bool_dtype(dtype):
                inferred_data_types[col_str] = "boolean"
                categorical_columns.append(col_str)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                inferred_data_types[col_str] = "datetime"
                categorical_columns.append(col_str)
            else:
                inferred_data_types[col_str] = "categorical"
                categorical_columns.append(col_str)

        return DatasetSummary(
            row_count=row_count,
            column_count=column_count,
            column_names=column_names,
            inferred_data_types=inferred_data_types,
            missing_value_count=missing_value_count,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns
        )
