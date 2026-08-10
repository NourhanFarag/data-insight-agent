import math
from typing import Union, List, Dict
import pandas as pd
from app.models.analysis import AnalysisStep, AnalysisResult, AnalysisOperation
from app.core.exceptions import AnalysisExecutionError

class AnalysisExecutor:
    def execute(self, df: pd.DataFrame, step: AnalysisStep, result_id: str = "result_1") -> AnalysisResult:
        """Executes a single deterministic AnalysisStep on the provided DataFrame.

        Validates columns, types, parameters, and performs the operation.
        """
        op = step.operation

        # Dispatch execution passing the deterministic result_id
        if op == AnalysisOperation.COUNT:
            return self._execute_count(df, step, result_id)
        elif op == AnalysisOperation.MEAN:
            return self._execute_mean(df, step, result_id)
        elif op == AnalysisOperation.MEDIAN:
            return self._execute_median(df, step, result_id)
        elif op == AnalysisOperation.MIN:
            return self._execute_min(df, step, result_id)
        elif op == AnalysisOperation.MAX:
            return self._execute_max(df, step, result_id)
        elif op == AnalysisOperation.SUM:
            return self._execute_sum(df, step, result_id)
        elif op == AnalysisOperation.STD:
            return self._execute_std(df, step, result_id)
        elif op == AnalysisOperation.MISSING_VALUES:
            return self._execute_missing_values(df, step, result_id)
        elif op == AnalysisOperation.UNIQUE_COUNT:
            return self._execute_unique_count(df, step, result_id)
        elif op == AnalysisOperation.TOP_VALUES:
            return self._execute_top_values(df, step, result_id)
        elif op == AnalysisOperation.GROUP_BY_MEAN:
            return self._execute_group_by_mean(df, step, result_id)
        elif op == AnalysisOperation.GROUP_BY_COUNT:
            return self._execute_group_by_count(df, step, result_id)
        elif op == AnalysisOperation.CORRELATION:
            return self._execute_correlation(df, step, result_id)
        else:
            raise AnalysisExecutionError(f"Unsupported operation: {op}")

    def _validate_column_exists(self, df: pd.DataFrame, col: str) -> None:
        if col not in df.columns:
            raise AnalysisExecutionError(f"Column '{col}' does not exist in dataset.")

    def _validate_numeric_column(self, df: pd.DataFrame, col: str, op: AnalysisOperation) -> None:
        self._validate_column_exists(df, col)
        dtype = df[col].dtype
        if not (pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)):
            raise AnalysisExecutionError(
                f"Column '{col}' must be numeric for operation {op.value}. "
                f"Inferred column type is {dtype}."
            )

    def _clean_numeric_result(self, val) -> Union[float, int, None]:
        """Convert numpy types to native Python types and handle NaN/Inf."""
        if pd.isna(val):
            return None
        if hasattr(val, "item"):
            val = val.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val

    def _execute_count(self, df: pd.DataFrame, step: AnalysisStep, result_id: str) -> AnalysisResult:
        if step.column is None:
            # COUNT without a column = total DataFrame rows.
            result_val = len(df)
            desc = f"Counted total rows in dataset. Result: {result_val}."
            return AnalysisResult(
                result_id=result_id,
                source_step_id=step.step_id,
                operation=AnalysisOperation.COUNT,
                target_columns=[],
                computed_result=result_val,
                description=desc
            )
        else:
            # COUNT with a column = number of non-null values in that column.
            col = step.column
            self._validate_column_exists(df, col)
            result_val = int(df[col].count())
            desc = f"Counted non-null values in column '{col}'. Result: {result_val}."
            return AnalysisResult(
                result_id=result_id,
                source_step_id=step.step_id,
                operation=AnalysisOperation.COUNT,
                target_columns=[col],
                computed_result=result_val,
                description=desc
            )

    def _execute_mean(self, df: pd.DataFrame, step: AnalysisStep, result_id: str) -> AnalysisResult:
        if not step.column:
            raise AnalysisExecutionError("Operation MEAN requires a target column.")
        col = step.column
        self._validate_numeric_column(df, col, AnalysisOperation.MEAN)

        mean_val = df[col].mean()
        result_val = self._clean_numeric_result(mean_val)

        return AnalysisResult(
            result_id=result_id,
            source_step_id=step.step_id,
            operation=AnalysisOperation.MEAN,
            target_columns=[col],
            computed_result=result_val,
            description=f"Computed mean on column '{col}'. Result: {result_val}."
        )

    def _execute_median(self, df: pd.DataFrame, step: AnalysisStep, result_id: str) -> AnalysisResult:
        if not step.column:
            raise AnalysisExecutionError("Operation MEDIAN requires a target column.")
        col = step.column
        self._validate_numeric_column(df, col, AnalysisOperation.MEDIAN)

        med_val = df[col].median()
        result_val = self._clean_numeric_result(med_val)

        return AnalysisResult(
            result_id=result_id,
            source_step_id=step.step_id,
            operation=AnalysisOperation.MEDIAN,
            target_columns=[col],
            computed_result=result_val,
            description=f"Computed median on column '{col}'. Result: {result_val}."
        )

    def _execute_min(self, df: pd.DataFrame, step: AnalysisStep, result_id: str) -> AnalysisResult:
        if not step.column:
            raise AnalysisExecutionError("Operation MIN requires a target column.")
        col = step.column
        self._validate_column_exists(df, col)

        min_val = df[col].min()
        result_val = self._clean_numeric_result(min_val) if pd.api.types.is_numeric_dtype(df[col].dtype) else (str(min_val) if not pd.isna(min_val) else None)

        return AnalysisResult(
            result_id=result_id,
            source_step_id=step.step_id,
            operation=AnalysisOperation.MIN,
            target_columns=[col],
            computed_result=result_val,
            description=f"Computed minimum on column '{col}'. Result: {result_val}."
        )

    def _execute_max(self, df: pd.DataFrame, step: AnalysisStep, result_id: str) -> AnalysisResult:
        if not step.column:
            raise AnalysisExecutionError("Operation MAX requires a target column.")
        col = step.column
        self._validate_column_exists(df, col)

        max_val = df[col].max()
        result_val = self._clean_numeric_result(max_val) if pd.api.types.is_numeric_dtype(df[col].dtype) else (str(max_val) if not pd.isna(max_val) else None)

        return AnalysisResult(
            result_id=result_id,
            source_step_id=step.step_id,
            operation=AnalysisOperation.MAX,
            target_columns=[col],
            computed_result=result_val,
            description=f"Computed maximum on column '{col}'. Result: {result_val}."
        )

    def _execute_sum(self, df: pd.DataFrame, step: AnalysisStep, result_id: str) -> AnalysisResult:
        if not step.column:
            raise AnalysisExecutionError("Operation SUM requires a target column.")
        col = step.column
        self._validate_numeric_column(df, col, AnalysisOperation.SUM)

        sum_val = df[col].sum()
        result_val = self._clean_numeric_result(sum_val)

        return AnalysisResult(
            result_id=result_id,
            source_step_id=step.step_id,
            operation=AnalysisOperation.SUM,
            target_columns=[col],
            computed_result=result_val,
            description=f"Computed sum on column '{col}'. Result: {result_val}."
        )

    def _execute_std(self, df: pd.DataFrame, step: AnalysisStep, result_id: str) -> AnalysisResult:
        if not step.column:
            raise AnalysisExecutionError("Operation STD requires a target column.")
        col = step.column
        self._validate_numeric_column(df, col, AnalysisOperation.STD)

        # Uses sample standard deviation with degrees of freedom ddof=1 by default in pandas
        std_val = df[col].std(ddof=1)
        result_val = self._clean_numeric_result(std_val)

        return AnalysisResult(
            result_id=result_id,
            source_step_id=step.step_id,
            operation=AnalysisOperation.STD,
            target_columns=[col],
            computed_result=result_val,
            description=f"Computed sample standard deviation (ddof=1) on column '{col}'. Result: {result_val}."
        )

    def _execute_missing_values(self, df: pd.DataFrame, step: AnalysisStep, result_id: str) -> AnalysisResult:
        if not step.column:
            raise AnalysisExecutionError("Operation MISSING_VALUES requires a target column.")
        col = step.column
        self._validate_column_exists(df, col)

        null_count = int(df[col].isnull().sum())

        return AnalysisResult(
            result_id=result_id,
            source_step_id=step.step_id,
            operation=AnalysisOperation.MISSING_VALUES,
            target_columns=[col],
            computed_result=null_count,
            description=f"Computed missing value count on column '{col}'. Result: {null_count}."
        )

    def _execute_unique_count(self, df: pd.DataFrame, step: AnalysisStep, result_id: str) -> AnalysisResult:
        if not step.column:
            raise AnalysisExecutionError("Operation UNIQUE_COUNT requires a target column.")
        col = step.column
        self._validate_column_exists(df, col)

        uniq_count = int(df[col].nunique())

        return AnalysisResult(
            result_id=result_id,
            source_step_id=step.step_id,
            operation=AnalysisOperation.UNIQUE_COUNT,
            target_columns=[col],
            computed_result=uniq_count,
            description=f"Computed unique values count on column '{col}'. Result: {uniq_count}."
        )

    def _execute_top_values(self, df: pd.DataFrame, step: AnalysisStep, result_id: str) -> AnalysisResult:
        if not step.column:
            raise AnalysisExecutionError("Operation TOP_VALUES requires a target column.")
        col = step.column
        self._validate_column_exists(df, col)

        # Validate limit (must be between 1 and 100)
        limit = step.limit
        if limit is None:
            limit = 10
        elif not isinstance(limit, int) or limit <= 0 or limit > 100:
            raise AnalysisExecutionError(f"Operation TOP_VALUES requires a limit between 1 and 100. Received: {limit}")

        value_counts = df[col].value_counts().head(limit)

        # Format as string keys for JSON compliance
        result_dict = {str(k): int(v) for k, v in value_counts.items()}

        return AnalysisResult(
            result_id=result_id,
            source_step_id=step.step_id,
            operation=AnalysisOperation.TOP_VALUES,
            target_columns=[col],
            computed_result=result_dict,
            description=f"Computed top {limit} values on column '{col}'."
        )

    def _execute_group_by_mean(self, df: pd.DataFrame, step: AnalysisStep, result_id: str) -> AnalysisResult:
        if not step.column:
            raise AnalysisExecutionError("Operation GROUP_BY_MEAN requires a target column (column).")
        if not step.group_by:
            raise AnalysisExecutionError("Operation GROUP_BY_MEAN requires a grouping column (group_by).")

        col = step.column
        gb_col = step.group_by

        self._validate_numeric_column(df, col, AnalysisOperation.GROUP_BY_MEAN)
        self._validate_column_exists(df, gb_col)

        grouped = df.groupby(gb_col)[col].mean()
        result_dict = {}
        for k, v in grouped.items():
            cleaned_val = self._clean_numeric_result(v)
            result_dict[str(k)] = cleaned_val

        return AnalysisResult(
            result_id=result_id,
            source_step_id=step.step_id,
            operation=AnalysisOperation.GROUP_BY_MEAN,
            target_columns=[col],
            grouping_column=gb_col,
            computed_result=result_dict,
            description=f"Computed mean of '{col}' grouped by '{gb_col}'."
        )

    def _execute_group_by_count(self, df: pd.DataFrame, step: AnalysisStep, result_id: str) -> AnalysisResult:
        if not step.group_by:
            raise AnalysisExecutionError("Operation GROUP_BY_COUNT requires a grouping column (group_by).")

        gb_col = step.group_by
        self._validate_column_exists(df, gb_col)

        col = step.column
        if col:
            self._validate_column_exists(df, col)
            grouped = df.groupby(gb_col)[col].count()
        else:
            grouped = df.groupby(gb_col).size()

        result_dict = {str(k): int(v) for k, v in grouped.items()}

        target = [col] if col else []
        return AnalysisResult(
            result_id=result_id,
            source_step_id=step.step_id,
            operation=AnalysisOperation.GROUP_BY_COUNT,
            target_columns=target,
            grouping_column=gb_col,
            computed_result=result_dict,
            description=f"Computed count of rows grouped by '{gb_col}'" + (f" using non-nulls of '{col}'." if col else ".")
        )

    def _execute_correlation(self, df: pd.DataFrame, step: AnalysisStep, result_id: str) -> AnalysisResult:
        col = step.column
        col2 = step.second_column

        if not col or not col2:
            raise AnalysisExecutionError("Operation CORRELATION requires both 'column' and 'second_column' specified.")

        self._validate_numeric_column(df, col, AnalysisOperation.CORRELATION)
        self._validate_numeric_column(df, col2, AnalysisOperation.CORRELATION)

        corr_val = df[col].corr(df[col2])
        result_val = self._clean_numeric_result(corr_val)

        return AnalysisResult(
            result_id=result_id,
            source_step_id=step.step_id,
            operation=AnalysisOperation.CORRELATION,
            target_columns=[col, col2],
            computed_result=result_val,
            description=f"Computed Pearson correlation coefficient between '{col}' and '{col2}'. Result: {result_val}."
        )
