from app.providers.base import BaseProvider
from app.models.responses import DatasetSummary
from app.models.analysis import (
    AnalysisPlan,
    AnalysisStep,
    ProviderReport,
    AnalysisResult,
    AnalysisOperation,
    ConfidenceLevel,
    Finding,
    Recommendation,
    RecommendationPriority
)

class MockProvider(BaseProvider):
    async def create_analysis_plan(self, question: str, summary: DatasetSummary) -> AnalysisPlan:
        """Generates a deterministic analysis plan based on the columns present in the dataset."""
        steps = []

        # 1. COUNT
        steps.append(
            AnalysisStep(
                step_id=f"step_{len(steps) + 1}",
                operation=AnalysisOperation.COUNT,
                reason="Determine total row count of dataset for context."
            )
        )

        # 2. MEAN (on the first numeric column if available)
        if summary.numeric_columns:
            target_num = summary.numeric_columns[0]
            steps.append(
                AnalysisStep(
                    step_id=f"step_{len(steps) + 1}",
                    operation=AnalysisOperation.MEAN,
                    column=target_num,
                    reason=f"Calculate average value for numeric field '{target_num}'."
                )
            )

            # 3. GROUP_BY_MEAN (if categorical columns also exist)
            if summary.categorical_columns:
                target_cat = summary.categorical_columns[0]
                steps.append(
                    AnalysisStep(
                        step_id=f"step_{len(steps) + 1}",
                        operation=AnalysisOperation.GROUP_BY_MEAN,
                        column=target_num,
                        group_by=target_cat,
                        reason=f"Analyze average '{target_num}' breakdown across groups of '{target_cat}'."
                    )
                )

        # 4. TOP_VALUES (on the first categorical column if available)
        if summary.categorical_columns:
            target_cat = summary.categorical_columns[0]
            steps.append(
                AnalysisStep(
                    step_id=f"step_{len(steps) + 1}",
                    operation=AnalysisOperation.TOP_VALUES,
                    column=target_cat,
                    limit=5,
                    reason=f"Identify most frequent categories in '{target_cat}'."
                )
            )

        return AnalysisPlan(
            objective=f"Address dataset question: '{question}' using deterministic steps.",
            steps=steps
        )

    async def repair_analysis_plan(
        self,
        question: str,
        dataset_summary: DatasetSummary,
        invalid_plan: AnalysisPlan,
        validation_feedback: str,
    ) -> AnalysisPlan:
        """Deterministically repairs an invalid AnalysisPlan."""
        if "fail_repair" in question.lower() or "fail_repair" in validation_feedback.lower():
            return AnalysisPlan(
                objective="Fail repair intentionally",
                steps=[
                    AnalysisStep(
                        step_id="step_1",
                        operation=AnalysisOperation.TOP_VALUES,
                        column="invalid_col_name_not_existing",
                        reason="Repair attempt that fails validation"
                    )
                ]
            )
        return await self.create_analysis_plan(question, dataset_summary)

    async def generate_report(
        self, question: str, summary: DatasetSummary, results: list[AnalysisResult]
    ) -> ProviderReport:
        """Generates a deterministic report summarizing the executed analysis results."""
        findings = []

        # Helper to find result by operation
        def find_res_by_op(op):
            return [r for r in results if r.operation == op]

        count_results = find_res_by_op(AnalysisOperation.COUNT)
        if count_results:
            findings.append(
                Finding(
                    id=f"finding_{len(findings) + 1}",
                    title="Dataset Scale Analysis",
                    explanation=f"Based on total row calculations, the dataset contains {count_results[0].computed_result} active rows.",
                    evidence_refs=[count_results[0].result_id],
                    confidence=ConfidenceLevel.HIGH
                )
            )

        mean_results = find_res_by_op(AnalysisOperation.MEAN)
        if mean_results:
            findings.append(
                Finding(
                    id=f"finding_{len(findings) + 1}",
                    title="Statistical Average Analysis",
                    explanation=f"Averages indicate a baseline of {mean_results[0].computed_result} for target metric {mean_results[0].target_columns[0]}.",
                    evidence_refs=[mean_results[0].result_id],
                    confidence=ConfidenceLevel.HIGH
                )
            )

        gb_mean_results = find_res_by_op(AnalysisOperation.GROUP_BY_MEAN)
        if gb_mean_results:
            findings.append(
                Finding(
                    id=f"finding_{len(findings) + 1}",
                    title="Group Performance Breakdown",
                    explanation=f"Breakdown analysis of {gb_mean_results[0].target_columns[0]} by {gb_mean_results[0].grouping_column} shows: {gb_mean_results[0].computed_result}.",
                    evidence_refs=[gb_mean_results[0].result_id],
                    confidence=ConfidenceLevel.HIGH
                )
            )

        top_results = find_res_by_op(AnalysisOperation.TOP_VALUES)
        if top_results:
            findings.append(
                Finding(
                    id=f"finding_{len(findings) + 1}",
                    title="Top Category Distribution",
                    explanation=f"The most frequent categories are distributed as follows: {top_results[0].computed_result}.",
                    evidence_refs=[top_results[0].result_id],
                    confidence=ConfidenceLevel.HIGH
                )
            )

        # Recommendations
        recommendations = []
        if findings:
            recommendations.append(
                Recommendation(
                    id=f"recommendation_{len(recommendations) + 1}",
                    priority=RecommendationPriority.HIGH,
                    action="Implement targeted resource allocation based on group breakdowns.",
                    rationale=f"Analyzing specific performance variances (referenced in {findings[0].id}) will help direct funding/effort.",
                    finding_refs=[findings[0].id]
                )
            )
            if len(findings) > 1:
                recommendations.append(
                    Recommendation(
                        id=f"recommendation_{len(recommendations) + 1}",
                        priority=RecommendationPriority.MEDIUM,
                        action="Establish continuous baseline monitoring.",
                        rationale=f"Baseline statistics from {findings[1].id} provide initial targets for dashboard KPIs.",
                        finding_refs=[findings[1].id]
                    )
                )

        # Fallback if no findings were generated
        if not findings:
            fallback_res_id = results[0].result_id if results else "result_1"
            findings.append(
                Finding(
                    id="finding_1",
                    title="Empty Results Analysis",
                    explanation="No analytical steps were successfully evaluated.",
                    evidence_refs=[fallback_res_id],
                    confidence=ConfidenceLevel.LOW
                )
            )
            recommendations.append(
                Recommendation(
                    id="recommendation_1",
                    priority=RecommendationPriority.LOW,
                    action="Perform a standard full dataset run.",
                    rationale="Empty results limit our ability to recommend active operations.",
                    finding_refs=["finding_1"]
                )
            )

        limitations = [
            "This report is generated using a mock provider model and is meant for verification of data pipeline pathways."
        ]

        return ProviderReport(
            findings=findings,
            limitations=limitations,
            recommendations=recommendations
        )
