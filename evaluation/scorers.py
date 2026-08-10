import re
import math
from typing import List, Dict, Any, Set
from app.models.analysis import AnalysisPlan, AnalysisResult, ProviderReport
from app.models.responses import DatasetSummary
from app.services.plan_validator import PlanValidator
from app.services.grounding_validator import GroundingValidator
from app.core.exceptions import PlanValidationError, GroundingValidationError
from evaluation.models import EvaluationCase, PlannerScores, GroundingScores, ExpectedResultCheck

def _extract_all_numbers(text: str) -> Set[float]:
    """Helper to extract numbers from explanation prose, ignoring markdown links/refs."""
    # Remove references like result_1, finding_1, step_1
    cleaned = re.sub(r"\b(?:result|finding|step)_[0-9]+\b", "", text, flags=re.IGNORECASE)
    # Match decimal numbers, percentages, etc.
    matches = re.findall(r"\b\d+(?:\.\d+)?%?\b", cleaned)
    nums = set()
    for m in matches:
        is_pct = m.endswith("%")
        val_str = m.rstrip("%")
        try:
            val = float(val_str)
            nums.add(val)
            if is_pct:
                nums.add(val / 100.0)
        except ValueError:
            continue
    return nums

def _collect_result_values(val: Any) -> Set[float]:
    """Recursively collect all numeric values present in a computed result."""
    values = set()
    if isinstance(val, (int, float)):
        values.add(float(val))
    elif isinstance(val, dict):
        for k, v in val.items():
            values.update(_collect_result_values(v))
            # Also try parsing key as numeric (e.g. top value count categories that are numeric)
            try:
                values.add(float(k))
            except ValueError:
                pass
    elif isinstance(val, (list, tuple)):
        for item in val:
            values.update(_collect_result_values(item))
    return values

def score_plan(proposed_plan: AnalysisPlan, case: EvaluationCase, summary: DatasetSummary) -> PlannerScores:
    """Computes structural and safety scores for the proposed plan."""
    schema_valid = len(proposed_plan.steps) > 0

    # Validate using the production validator
    validator = PlanValidator()
    try:
        validator.validate(proposed_plan, summary)
        plan_valid = True
    except PlanValidationError:
        plan_valid = False

    # Required operation recall
    if not case.required_operations:
        recall = 1.0
    else:
        matched = 0
        for req_op in case.required_operations:
            # Check if there is any step matching this operation
            if any(step.operation == req_op for step in proposed_plan.steps):
                matched += 1
        recall = matched / len(case.required_operations)

    # Irrelevant operation rate
    if not proposed_plan.steps:
        irrelevant_rate = 0.0
    else:
        irrelevant_steps = 0
        for step in proposed_plan.steps:
            is_req = step.operation in case.required_operations
            is_acc = step.operation in case.acceptable_operations
            if not is_req and not is_acc:
                irrelevant_steps += 1
        irrelevant_rate = irrelevant_steps / len(proposed_plan.steps)

    # Invalid column attempts
    invalid_cols = 0
    for step in proposed_plan.steps:
        if step.column and step.column not in summary.column_names:
            invalid_cols += 1
        if step.second_column and step.second_column not in summary.column_names:
            invalid_cols += 1

    # Success definition: valid schema/plan and recall above threshold (85%)
    planner_success = schema_valid and plan_valid and (recall >= 0.85)

    return PlannerScores(
        schema_valid=schema_valid,
        plan_valid=plan_valid,
        required_operation_recall=recall,
        irrelevant_operation_rate=irrelevant_rate,
        invalid_column_attempts=invalid_cols,
        planner_success=planner_success
    )

def verify_execution(results: List[AnalysisResult], case: EvaluationCase) -> bool:
    """Checks computed results against ground truth expected checks."""
    for check in case.expected_result_checks:
        # Find matching result
        matching_res = None
        for res in results:
            if res.operation == check.operation:
                # If column is specified, match it
                if check.column and check.column not in res.target_columns:
                    continue
                if check.group_by and res.grouping_column != check.group_by:
                    continue
                matching_res = res
                break

        if not matching_res:
            return False

        actual = matching_res.computed_result
        expected = check.expected_value
        tol = check.tolerance

        if not _compare_values(actual, expected, tol):
            return False

    return True

def _compare_values(actual: Any, expected: Any, tol: Optional[float]) -> bool:
    """Helper to compare execution outputs with ground truth."""
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if tol is not None:
            return abs(actual - expected) <= tol
        return actual == expected

    if isinstance(actual, dict) and isinstance(expected, dict):
        if len(actual) != len(expected):
            return False
        for k, v in expected.items():
            if k not in actual:
                return False
            if not _compare_values(actual[k], v, tol):
                return False
        return True

    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return False
        for a, e in zip(actual, expected):
            if not _compare_values(a, e, tol):
                return False
        return True

    return actual == expected

def score_report(report: ProviderReport, results: List[AnalysisResult], case: EvaluationCase) -> GroundingScores:
    """Scores grounding correctness and scans for causal/unsupported claims."""
    # 1. Structural Grounding Check
    g_validator = GroundingValidator()
    try:
        g_validator.validate(report, results)
        structurally_grounded = True
    except GroundingValidationError:
        structurally_grounded = False

    # 2. Unsupported Numeric Claims Scan
    # Collect all numeric values generated by the actual executed computations
    computed_numbers = set()
    for res in results:
        computed_numbers.update(_collect_result_values(res.computed_result))

    unsupported_numeric_flags = 0
    for finding in report.findings:
        text_nums = _extract_all_numbers(finding.explanation)
        for num in text_nums:
            # Ignore basic low integer counters (0 to 5) to prevent false positives on sentence prose
            if num in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
                continue
            # Check if this number is close to any computed value
            matched = False
            for comp_num in computed_numbers:
                if abs(num - comp_num) < 1e-4:
                    matched = True
                    break
                # Check percentage equivalence (e.g. 20% in text matching 0.20 in results)
                if abs(num / 100.0 - comp_num) < 1e-4:
                    matched = True
                    break
                if abs(num * 100.0 - comp_num) < 1e-4:
                    matched = True
                    break
            if not matched:
                unsupported_numeric_flags += 1

    # 3. Causal Claim Scan
    causal_flags = 0
    # Causal scanning is only relevant if dataset has descriptive/correlation tag
    if any(tag in case.tags for tag in ["correlation", "weak_correlation", "descriptive"]):
        causal_keywords = [
            r"\bcauses\b", r"\bcaused\b", r"\bleads to\b",
            r"\bresults in\b", r"\bbecause of\b", r"\bdrives\b"
        ]
        text_to_scan = []
        for finding in report.findings:
            text_to_scan.append(finding.explanation.lower())
        for rec in report.recommendations:
            text_to_scan.append(rec.rationale.lower())

        for text in text_to_scan:
            # Strip structural references like "because of finding_1" to avoid false positives on linking prose
            cleaned_text = re.sub(r"\bbecause of\s+(?:finding|result)_[0-9]+\b", "", text, flags=re.IGNORECASE)
            for kw in causal_keywords:
                if re.search(kw, cleaned_text):
                    causal_flags += 1

    return GroundingScores(
        structurally_grounded=structurally_grounded,
        unsupported_numeric_claim_flags=unsupported_numeric_flags,
        causal_claim_flags=causal_flags
    )
