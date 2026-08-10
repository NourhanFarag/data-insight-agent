from app.models.analysis import AnalysisPlan, AnalysisOperation
from app.models.responses import DatasetSummary
from app.core.exceptions import PlanValidationError
from app.config import settings

class PlanValidator:
    @staticmethod
    def validate(plan: AnalysisPlan, summary: DatasetSummary) -> None:
        """Validates that the proposed analysis plan is structurally sound, safe,

        corresponds to columns in the summary, and uses correct datatypes.
        """
        if not plan.steps:
            raise PlanValidationError("Analysis plan cannot be empty.")

        if len(plan.steps) > settings.MAX_ANALYSIS_STEPS:
            raise PlanValidationError(f"Analysis plan step count ({len(plan.steps)}) exceeds configured maximum limit of {settings.MAX_ANALYSIS_STEPS}.")

        step_ids = set()
        seen_steps = set()

        for idx, step in enumerate(plan.steps):
            step_id = step.step_id

            # Validate ID formatting
            if not step_id or not step_id.strip():
                raise PlanValidationError(f"Step at index {idx} contains an empty 'step_id'.")
            if step_id in step_ids:
                raise PlanValidationError(f"Duplicate step ID found: '{step_id}'. Each step must have a unique ID.")
            step_ids.add(step_id)

            # Validate duplicate equivalent steps
            step_sig = (
                step.operation,
                step.column,
                step.group_by,
                step.second_column,
                step.limit
            )
            if step_sig in seen_steps:
                raise PlanValidationError(f"Duplicate equivalent step detected at '{step_id}'.")
            seen_steps.add(step_sig)

            op = step.operation

            # COUNT safety checks
            if op == AnalysisOperation.COUNT:
                if step.group_by or step.second_column or step.limit:
                    raise PlanValidationError(f"Operation COUNT in '{step_id}' cannot specify 'group_by', 'second_column', or 'limit'.")
                if step.column and step.column not in summary.column_names:
                    raise PlanValidationError(f"Column '{step.column}' in step '{step_id}' does not exist in dataset.")

            # Numeric aggregation checks
            elif op in (AnalysisOperation.MEAN, AnalysisOperation.MEDIAN, AnalysisOperation.SUM, AnalysisOperation.STD):
                if not step.column:
                    raise PlanValidationError(f"Operation {op.value} in '{step_id}' requires a 'column' parameter.")
                if step.group_by or step.second_column or step.limit:
                    raise PlanValidationError(f"Operation {op.value} in '{step_id}' cannot specify 'group_by', 'second_column', or 'limit'.")
                if step.column not in summary.numeric_columns:
                    if step.column not in summary.column_names:
                        raise PlanValidationError(f"Column '{step.column}' in step '{step_id}' does not exist in dataset.")
                    raise PlanValidationError(f"Operation {op.value} on step '{step_id}' requires numeric column. Column '{step.column}' is not numeric.")

            # General single-column checks
            elif op in (AnalysisOperation.MIN, AnalysisOperation.MAX, AnalysisOperation.MISSING_VALUES, AnalysisOperation.UNIQUE_COUNT):
                if not step.column:
                    raise PlanValidationError(f"Operation {op.value} in '{step_id}' requires a 'column' parameter.")
                if step.group_by or step.second_column or step.limit:
                    raise PlanValidationError(f"Operation {op.value} in '{step_id}' cannot specify 'group_by', 'second_column', or 'limit'.")
                if step.column not in summary.column_names:
                    raise PlanValidationError(f"Column '{step.column}' in step '{step_id}' does not exist in dataset.")

            # TOP_VALUES check
            elif op == AnalysisOperation.TOP_VALUES:
                if not step.column:
                    raise PlanValidationError(f"Operation TOP_VALUES in '{step_id}' requires a 'column' parameter.")
                if step.group_by or step.second_column:
                    raise PlanValidationError(f"Operation TOP_VALUES in '{step_id}' cannot specify 'group_by' or 'second_column'.")
                if step.column not in summary.column_names:
                    raise PlanValidationError(f"Column '{step.column}' in step '{step_id}' does not exist in dataset.")
                if step.limit is not None:
                    if not isinstance(step.limit, int) or step.limit <= 0 or step.limit > 100:
                        raise PlanValidationError(f"Operation TOP_VALUES in '{step_id}' contains an invalid 'limit' of {step.limit}. Must be an integer between 1 and 100.")

            # GROUP_BY_MEAN check
            elif op == AnalysisOperation.GROUP_BY_MEAN:
                if not step.column or not step.group_by:
                    raise PlanValidationError(f"Operation GROUP_BY_MEAN in '{step_id}' requires both 'column' and 'group_by' parameters.")
                if step.second_column or step.limit:
                    raise PlanValidationError(f"Operation GROUP_BY_MEAN in '{step_id}' cannot specify 'second_column' or 'limit'.")
                if step.column not in summary.numeric_columns:
                    if step.column not in summary.column_names:
                        raise PlanValidationError(f"Column '{step.column}' in step '{step_id}' does not exist in dataset.")
                    raise PlanValidationError(f"Operation GROUP_BY_MEAN target column '{step.column}' in step '{step_id}' must be numeric.")
                if step.group_by not in summary.column_names:
                    raise PlanValidationError(f"Grouping column '{step.group_by}' in step '{step_id}' does not exist in dataset.")

            # GROUP_BY_COUNT check
            elif op == AnalysisOperation.GROUP_BY_COUNT:
                if not step.group_by:
                    raise PlanValidationError(f"Operation GROUP_BY_COUNT in '{step_id}' requires a 'group_by' parameter.")
                if step.second_column or step.limit:
                    raise PlanValidationError(f"Operation GROUP_BY_COUNT in '{step_id}' cannot specify 'second_column' or 'limit'.")
                if step.group_by not in summary.column_names:
                    raise PlanValidationError(f"Grouping column '{step.group_by}' in step '{step_id}' does not exist in dataset.")
                if step.column and step.column not in summary.column_names:
                    raise PlanValidationError(f"Target column '{step.column}' in step '{step_id}' does not exist in dataset.")

            # CORRELATION check
            elif op == AnalysisOperation.CORRELATION:
                if not step.column or not step.second_column:
                    raise PlanValidationError(f"Operation CORRELATION in '{step_id}' requires both 'column' and 'second_column' parameters.")
                if step.group_by or step.limit:
                    raise PlanValidationError(f"Operation CORRELATION in '{step_id}' cannot specify 'group_by' or 'limit'.")
                if step.column not in summary.numeric_columns:
                    if step.column not in summary.column_names:
                        raise PlanValidationError(f"Column '{step.column}' in step '{step_id}' does not exist in dataset.")
                    raise PlanValidationError(f"Operation CORRELATION column '{step.column}' in step '{step_id}' must be numeric.")
                if step.second_column not in summary.numeric_columns:
                    if step.second_column not in summary.column_names:
                        raise PlanValidationError(f"Column '{step.second_column}' in step '{step_id}' does not exist in dataset.")
                    raise PlanValidationError(f"Operation CORRELATION second_column '{step.second_column}' in step '{step_id}' must be numeric.")
