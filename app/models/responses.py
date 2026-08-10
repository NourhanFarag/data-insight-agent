from typing import List, Dict
from pydantic import BaseModel, Field
from app.models.analysis import AnalysisPlan, AnalysisResult, Finding, Recommendation

class DatasetSummary(BaseModel):
    row_count: int = Field(..., description="Total number of rows in the dataset")
    column_count: int = Field(..., description="Total number of columns in the dataset")
    column_names: List[str] = Field(..., description="List of all column names in order")
    inferred_data_types: Dict[str, str] = Field(..., description="Mapping of column names to clean string data types (numeric, categorical, etc.)")
    missing_value_count: Dict[str, int] = Field(..., description="Mapping of column names to count of missing values")
    numeric_columns: List[str] = Field(..., description="Columns containing numeric data (int, float)")
    categorical_columns: List[str] = Field(..., description="Columns containing text or categorical/unstructured data")

class AnalysisResponse(BaseModel):
    question: str = Field(..., description="The original validated user request question")
    dataset_summary: DatasetSummary = Field(..., description="The verified dataset summary")
    analysis_plan: AnalysisPlan = Field(..., description="The validated analysis plan executed on the dataset")
    analysis_results: List[AnalysisResult] = Field(..., description="The deterministic results of the executed analysis steps")
    findings: List[Finding] = Field(..., description="Verified findings grounded to calculation results")
    limitations: List[str] = Field(..., description="Identified limitations or constraints of the data")
    recommendations: List[Recommendation] = Field(..., description="Traceable recommendations linked back to findings")
    plan_repair_attempted: bool = Field(default=False, description="Whether plan repair was attempted")
    plan_repair_succeeded: bool = Field(default=False, description="Whether plan repair succeeded")
    report_repair_attempted: bool = Field(default=False, description="Whether report repair was attempted")
    report_repair_succeeded: bool = Field(default=False, description="Whether report repair succeeded")
