import re
import math
from typing import List, Dict, Any, Set
from app.models.analysis import AnalysisPlan, AnalysisResult, ProviderReport, AnalysisOperation
from app.models.responses import DatasetSummary
from app.services.plan_validator import PlanValidator
from app.services.grounding_validator import GroundingValidator
from app.core.exceptions import PlanValidationError, GroundingValidationError
from evaluation.models import EvaluationCase, PlannerScores, GroundingScores, ExpectedResultCheck, ExecutionCheckDiagnostic

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

    # Required operation recall (exact vs semantic)
    exact_matched = 0
    semantic_matched = 0
    if not case.required_operations:
        recall = 1.0
        semantic_recall = 1.0
    else:
        for req_op in case.required_operations:
            if any(step.operation == req_op for step in proposed_plan.steps):
                exact_matched += 1
                semantic_matched += 1
            else:
                equivs = case.equivalent_operations.get(req_op, []) if case.equivalent_operations else []
                if any(step.operation in equivs for step in proposed_plan.steps):
                    semantic_matched += 1
        recall = exact_matched / len(case.required_operations)
        semantic_recall = semantic_matched / len(case.required_operations)

    # Irrelevant operation rate
    if not proposed_plan.steps:
        irrelevant_rate = 0.0
    else:
        irrelevant_steps = 0
        for step in proposed_plan.steps:
            is_req = step.operation in case.required_operations
            is_acc = step.operation in case.acceptable_operations
            is_equiv = False
            if case.equivalent_operations:
                for req_op, equivs in case.equivalent_operations.items():
                    if req_op in case.required_operations and step.operation in equivs:
                        is_equiv = True
                        break
            if not is_req and not is_acc and not is_equiv:
                irrelevant_steps += 1
        irrelevant_rate = irrelevant_steps / len(proposed_plan.steps)

    # Invalid column attempts
    invalid_cols = 0
    for step in proposed_plan.steps:
        if step.column and step.column not in summary.column_names:
            invalid_cols += 1
        if step.second_column and step.second_column not in summary.column_names:
            invalid_cols += 1

    # Success definition: valid schema/plan and semantic recall above threshold (85%)
    planner_success = schema_valid and plan_valid and (semantic_recall >= 0.85)

    return PlannerScores(
        schema_valid=schema_valid,
        plan_valid=plan_valid,
        required_operation_recall=recall,
        irrelevant_operation_rate=irrelevant_rate,
        invalid_column_attempts=invalid_cols,
        planner_success=planner_success,
        semantic_operation_recall=semantic_recall
    )

def _find_matching_result(results: List[AnalysisResult], check: ExpectedResultCheck, case: EvaluationCase) -> tuple[AnalysisResult | None, str | None]:
    """Helper to find matching result (exact or equivalent) and return (matching_res, mismatch_reason)."""
    # Check if operation exists at all
    has_op = False
    for res in results:
        if res.operation == check.operation:
            has_op = True
            break
        if case.equivalent_operations and res.operation in case.equivalent_operations.get(check.operation, []):
            has_op = True
            break
            
    if not has_op:
        return None, "missing_expected_operation"

    # Try exact match first
    for res in results:
        if res.operation == check.operation:
            # Check column
            if check.column and check.column not in res.target_columns:
                continue
            # Check group_by
            if check.group_by and res.grouping_column != check.group_by:
                continue
            # Check second_column
            if check.second_column and check.second_column not in res.target_columns:
                continue
            return res, None

    # Try equivalent match
    if case.equivalent_operations:
        equiv_ops = case.equivalent_operations.get(check.operation, [])
        for res in results:
            if res.operation in equiv_ops:
                # Target column / grouping semantics correspond
                if check.operation == AnalysisOperation.TOP_VALUES and res.operation == AnalysisOperation.GROUP_BY_COUNT:
                    if check.column and res.grouping_column != check.column:
                        continue
                    return res, None

    # If we had the operation but target semantics didn't match: Check mismatch reason in precedence order
    op_matches = [res for res in results if res.operation == check.operation or (case.equivalent_operations and res.operation in case.equivalent_operations.get(check.operation, []))]
    
    # Column check
    if check.column:
        has_col = False
        for r in op_matches:
            if r.operation == check.operation and check.column in r.target_columns:
                has_col = True
            elif check.operation == AnalysisOperation.TOP_VALUES and r.operation == AnalysisOperation.GROUP_BY_COUNT and r.grouping_column == check.column:
                has_col = True
        if not has_col:
            return None, "column_mismatch"

    # Group_by check
    if check.group_by:
        has_gb = False
        for r in op_matches:
            if r.operation == check.operation and r.grouping_column == check.group_by:
                has_gb = True
        if not has_gb:
            return None, "group_by_mismatch"

    # Second column check
    if check.second_column:
        has_sc = False
        for r in op_matches:
            if r.operation == check.operation and check.second_column in r.target_columns:
                has_sc = True
        if not has_sc:
            return None, "second_column_mismatch"

    return None, "missing_expected_operation"

def verify_execution(results: List[AnalysisResult], case: EvaluationCase) -> bool:
    """Checks computed results against ground truth expected checks."""
    for check in case.expected_result_checks:
        matching_res, _ = _find_matching_result(results, check, case)
        if not matching_res:
            return False

        actual = matching_res.computed_result
        expected = check.expected_value
        tol = check.tolerance

        if not _compare_values(actual, expected, tol):
            return False

    return True

def _compare_values(actual: Any, expected: Any, tol: float | None) -> bool:
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


def _sanitize_string(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    s_lower = s.lower()
    # Check for prompt injection or raw CSV cell leak
    if ";" in s or "--" in s or "drop" in s_lower or "select" in s_lower or "union" in s_lower or "delete" in s_lower:
        return "<redacted adversarial value>"
    # Also check for suspicious quotes or length
    if len(s) > 50 or any(c in s for c in ["'", '"', ";", "--", "/*", "*/"]):
        return "<redacted categorical value>"
    return s


def _sanitize_value(val: Any, tags: List[str] | None = None, registry: dict | None = None) -> Any:
    """Helper to safely serialize diagnostic values (expected/actual), redacting CSV cell payloads."""
    if registry is None:
        registry = {}
    is_adversarial = tags is not None and "adversarial" in tags

    if val is None:
        return None
    if isinstance(val, (int, float, bool)):
        return val
    if isinstance(val, str):
        if is_adversarial:
            if val not in registry:
                registry[val] = f"<redacted category {len(registry) + 1}>"
            return registry[val]
        return _sanitize_string(val)
    if isinstance(val, dict):
        sanitized_dict = {}
        for k, v in val.items():
            sanitized_key = _sanitize_value(k, tags, registry)
            sanitized_val = _sanitize_value(v, tags, registry)
            sanitized_dict[sanitized_key] = sanitized_val
        return sanitized_dict
    if isinstance(val, list):
        return [_sanitize_value(v, tags, registry) for v in val]
    return "<redacted categorical value>"


def _sanitize_actual_value(actual: Any, expected: Any, matched: bool, tags: List[str] | None = None, registry: dict | None = None) -> Any:
    """Helper to safely serialize computed actual values, keeping CSV cell payloads redacted."""
    return _sanitize_value(actual, tags, registry)


def diagnose_execution(
    results: List[AnalysisResult] | None,
    case: EvaluationCase,
) -> List[ExecutionCheckDiagnostic]:
    """Inspects expected result checks and explains why each comparison passed or failed."""
    diagnostics = []
    registry = {}

    for check in case.expected_result_checks:
        if results is None:
            diagnostics.append(
                ExecutionCheckDiagnostic(
                    expected_operation=check.operation,
                    expected_column=check.column,
                    expected_group_by=check.group_by,
                    expected_second_column=check.second_column,
                    expected_value=_sanitize_value(check.expected_value, case.tags, registry),
                    matching_result_found=False,
                    comparison_outcome=False,
                    mismatch_reason="execution_not_reached",
                    actual_value=None
                )
            )
            continue

        matching_res, mismatch_reason = _find_matching_result(results, check, case)

        if not matching_res:
            diagnostics.append(
                ExecutionCheckDiagnostic(
                    expected_operation=check.operation,
                    expected_column=check.column,
                    expected_group_by=check.group_by,
                    expected_second_column=check.second_column,
                    expected_value=_sanitize_value(check.expected_value, case.tags, registry),
                    matching_result_found=False,
                    comparison_outcome=False,
                    mismatch_reason=mismatch_reason or "missing_expected_operation",
                    actual_value=None
                )
            )
            continue

        # Perform comparison
        actual = matching_res.computed_result
        expected = check.expected_value
        tol = check.tolerance
        comparison_outcome = _compare_values(actual, expected, tol)

        if not comparison_outcome:
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                if tol is not None:
                    if abs(actual - expected) > tol:
                        mismatch_reason = "numeric_tolerance_mismatch"
                    else:
                        mismatch_reason = "value_mismatch"
                else:
                    mismatch_reason = "value_mismatch"
            elif isinstance(actual, (dict, list)) and isinstance(expected, (dict, list)):
                if len(actual) != len(expected):
                    mismatch_reason = "result_shape_mismatch"
                else:
                    mismatch_reason = "value_mismatch"
            else:
                mismatch_reason = "value_mismatch"

        sanitized_actual = _sanitize_actual_value(actual, expected, comparison_outcome, case.tags, registry)
        diagnostics.append(
            ExecutionCheckDiagnostic(
                expected_operation=check.operation,
                expected_column=check.column,
                expected_group_by=check.group_by,
                expected_second_column=check.second_column,
                expected_value=_sanitize_value(check.expected_value, case.tags, registry),
                matching_result_found=True,
                comparison_outcome=comparison_outcome,
                mismatch_reason=mismatch_reason,
                actual_value=sanitized_actual
            )
        )

    return diagnostics
