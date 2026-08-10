from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.models.analysis import AnalysisOperation, AnalysisPlan

class ExpectedResultCheck(BaseModel):
    operation: AnalysisOperation
    column: Optional[str] = None
    group_by: Optional[str] = None
    second_column: Optional[str] = None
    expected_value: Any  # Can be float, int, str, dict, etc.
    tolerance: Optional[float] = None

class EvaluationCase(BaseModel):
    case_id: str
    dataset_path: str
    question: str
    required_operations: List[AnalysisOperation]
    acceptable_operations: List[AnalysisOperation] = Field(default_factory=list)
    prohibited_operations: List[AnalysisOperation] = Field(default_factory=list)
    expected_result_checks: List[ExpectedResultCheck]
    equivalent_operations: Dict[AnalysisOperation, List[AnalysisOperation]] = Field(default_factory=dict)
    required_finding_concepts: List[str] = Field(default_factory=list)
    prohibited_claims: List[str] = Field(default_factory=list)
    acceptable_recommendation_themes: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class PlannerScores(BaseModel):
    schema_valid: bool
    plan_valid: bool
    required_operation_recall: float
    irrelevant_operation_rate: float
    invalid_column_attempts: int
    planner_success: bool
    plan_repair_attempted: bool = False
    plan_repair_succeeded: bool = False
    semantic_operation_recall: float = 0.0

class GroundingScores(BaseModel):
    structurally_grounded: bool
    unsupported_numeric_claim_flags: int
    causal_claim_flags: int

class HumanScores(BaseModel):
    relevance: Optional[int] = None
    finding_quality: Optional[int] = None
    recommendation_usefulness: Optional[int] = None
    restraint: Optional[int] = None
    clarity: Optional[int] = None

class ExecutionCheckDiagnostic(BaseModel):
    expected_operation: AnalysisOperation
    expected_column: Optional[str] = None
    expected_group_by: Optional[str] = None
    expected_second_column: Optional[str] = None
    expected_value: Any = None
    matching_result_found: bool
    comparison_outcome: bool
    mismatch_reason: Optional[str] = None
    actual_value: Any = None

class EvaluationResult(BaseModel):
    case_id: str
    provider: str
    model: str
    planner_scores: PlannerScores
    execution_passed: bool
    grounding_scores: GroundingScores
    human_scores: Optional[HumanScores] = None
    latency_ms: Optional[float] = None
    error_category: Optional[str] = None  # e.g., "provider_unavailable", "timeout", "unsafe_plan", etc.
    final_success: bool
    selected_plan: Optional[AnalysisPlan] = None
    execution_diagnostics: Optional[List[ExecutionCheckDiagnostic]] = None
