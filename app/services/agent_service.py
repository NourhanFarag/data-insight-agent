import logging
import pandas as pd
from app.models.responses import DatasetSummary, AnalysisResponse
from app.models.analysis import AnalysisPlan, AnalysisResult
from app.services.dataset_inspector import DatasetInspector
from app.services.analysis_executor import AnalysisExecutor
from app.services.plan_validator import PlanValidator
from app.services.grounding_validator import GroundingValidator
from app.providers import get_provider
from app.core.exceptions import (
    DatasetValidationError,
    PlanValidationError,
    GroundingValidationError,
    ProviderError,
)

logger = logging.getLogger("app.services.agent_service")
logging.basicConfig(level=logging.INFO)

class DataInsightAgent:
    def __init__(self):
        self.executor = AnalysisExecutor()
        self.plan_validator = PlanValidator()
        self.grounding_validator = GroundingValidator()
        self.plan_repair_attempted = False
        self.plan_repair_succeeded = False

    async def analyze(self, df: pd.DataFrame, question: str) -> AnalysisResponse:
        """Orchestrates the entire planning-execution-grounding loop on the dataset.

        Does not reveal raw CSV data to the LLM.
        """
        self.plan_repair_attempted = False
        self.plan_repair_succeeded = False
        # 1. Validate question input
        if not question or not question.strip():
            raise DatasetValidationError("Question cannot be empty.", status_code=400)

        # 2. Inspect DataFrame (pure python/pandas)
        summary = DatasetInspector.inspect(df)
        logger.info(
            f"Dataset inspected: {summary.row_count} rows, "
            f"{summary.column_count} columns."
        )

        # 3. Retrieve configured provider
        provider = get_provider()
        provider_name = provider.__class__.__name__
        logger.info(f"Using AI provider: {provider_name}")

        # 4. Request an AnalysisPlan from the provider
        logger.info(f"Requesting analysis plan for question: '{question}'")
        plan = await provider.create_analysis_plan(question, summary)

        # 5. Validate the plan against dataset schemas/limits
        logger.info("Validating proposed analysis plan...")
        try:
            self.plan_validator.validate(plan, summary)
        except PlanValidationError as exc:
            exc.invalid_plan = plan
            logger.warning(f"Initial plan validation failed: {exc}. Attempting plan repair...")
            self.plan_repair_attempted = True

            # Sanitize feedback
            sanitized_feedback = str(exc)

            # Request repaired plan
            try:
                repaired_plan = await provider.repair_analysis_plan(
                    question=question,
                    dataset_summary=summary,
                    invalid_plan=plan,
                    validation_feedback=sanitized_feedback
                )
            except Exception:
                exc.invalid_plan = plan
                raise exc

            # Re-validate the repaired plan
            logger.info("Validating repaired analysis plan...")
            try:
                self.plan_validator.validate(repaired_plan, summary)
                plan = repaired_plan
            except PlanValidationError as repair_exc:
                repair_exc.invalid_plan = repaired_plan
                raise repair_exc

            logger.info("Plan repair succeeded.")
            self.plan_repair_succeeded = True

        # 6. Execute approved steps using deterministic AnalysisExecutor
        results = []
        logger.info(f"Executing {len(plan.steps)} approved analysis steps...")
        for idx, step in enumerate(plan.steps):
            res_id = f"result_{idx + 1}"
            res = self.executor.execute(df, step, result_id=res_id)
            results.append(res)

        # 7. Request structured report/interpretations from provider
        logger.info("Requesting structured analysis report interpretation...")
        try:
            report = await provider.generate_report(question, summary, results)
        except ProviderError as pe:
            pe.analysis_plan = plan
            pe.analysis_results = results
            raise pe
        except Exception as e:
            try:
                e.analysis_plan = plan
                e.analysis_results = results
            except Exception:
                pass
            raise e

        # 8. Validate report grounding references
        logger.info("Validating report grounding and citations...")
        try:
            self.grounding_validator.validate(report, results)
        except GroundingValidationError as gve:
            gve.analysis_plan = plan
            gve.analysis_results = results
            raise gve

        logger.info("Analysis completed successfully.")

        # 9. Assemble and return trusted response
        return AnalysisResponse(
            question=question,
            dataset_summary=summary,
            analysis_plan=plan,
            analysis_results=results,
            findings=report.findings,
            limitations=report.limitations,
            recommendations=report.recommendations,
            plan_repair_attempted=self.plan_repair_attempted,
            plan_repair_succeeded=self.plan_repair_succeeded
        )
