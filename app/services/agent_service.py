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

def _sanitize_validation_feedback(msg: str) -> str:
    if not msg:
        return "Unknown validation error"
    msg = msg[:300]
    msg_lower = msg.lower()
    suspicious = ["ignore", "system", "instruction", "select", "drop", "union", "delete", "insert", "secret", "whoami", "os.system"]
    if any(s in msg_lower for s in suspicious):
        return "Sanitized validation error due to security policy"
    if "redacted" in msg_lower:
        return "Sanitized validation error referencing redacted content"
    return msg

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
        try:
            plan = await provider.create_analysis_plan(question, summary)
        except Exception as e:
            e.failure_stage = "planning"
            raise e

        # 5. Validate the plan against dataset schemas/limits
        logger.info("Validating proposed analysis plan...")
        try:
            self.plan_validator.validate(plan, summary)
        except PlanValidationError as exc:
            exc.invalid_plan = plan
            exc.failure_stage = "plan_validation"
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
            except Exception as rep_err:
                rep_err.invalid_plan = plan
                rep_err.failure_stage = "plan_validation"
                raise rep_err

            # Re-validate the repaired plan
            logger.info("Validating repaired analysis plan...")
            try:
                self.plan_validator.validate(repaired_plan, summary)
                plan = repaired_plan
            except PlanValidationError as repair_exc:
                repair_exc.invalid_plan = repaired_plan
                repair_exc.failure_stage = "plan_validation"
                raise repair_exc

            logger.info("Plan repair succeeded.")
            self.plan_repair_succeeded = True

        # 6. Execute approved steps using deterministic AnalysisExecutor
        results = []
        logger.info(f"Executing {len(plan.steps)} approved analysis steps...")
        try:
            for idx, step in enumerate(plan.steps):
                res_id = f"result_{idx + 1}"
                res = self.executor.execute(df, step, result_id=res_id)
                results.append(res)
        except Exception as exc:
            exc.analysis_plan = plan
            exc.analysis_results = results
            exc.failure_stage = "execution"
            raise exc

        # 6.5. Sanitize results before they are passed to any provider or validation steps
        from app.core.safety import _sanitize_value
        sanitized_results = []
        for res in results:
            sanitized_res = res.model_copy(update={
                "computed_result": _sanitize_value(res.computed_result)
            })
            sanitized_results.append(sanitized_res)

        # 7. Request structured report/interpretations from provider
        logger.info("Requesting structured analysis report interpretation...")
        try:
            report = await provider.generate_report(question, summary, sanitized_results)
        except ProviderError as pe:
            pe.analysis_plan = plan
            pe.analysis_results = sanitized_results
            pe.failure_stage = "report_generation"
            raise pe
        except Exception as e:
            try:
                e.analysis_plan = plan
                e.analysis_results = sanitized_results
            except Exception:
                pass
            e.failure_stage = "report_generation"
            raise e

        # 8. Validate report grounding references
        logger.info("Validating report grounding and citations...")
        report_repair_attempted = False
        report_repair_succeeded = False
        try:
            self.grounding_validator.validate(report, sanitized_results)
        except GroundingValidationError as gve:
            logger.warning(f"Initial report grounding validation failed: {gve}. Attempting report repair...")
            report_repair_attempted = True
            sanitized_feedback = _sanitize_validation_feedback(str(gve))

            try:
                repaired_report = await provider.repair_report(
                    question=question,
                    dataset_summary=summary,
                    analysis_results=sanitized_results,
                    invalid_report=report,
                    validation_feedback=sanitized_feedback
                )
            except Exception as rep_err:
                rep_err.analysis_plan = plan
                rep_err.analysis_results = sanitized_results
                rep_err.report = report
                rep_err.failure_stage = "grounding_validation"
                rep_err.report_repair_attempted = report_repair_attempted
                rep_err.report_repair_succeeded = report_repair_succeeded
                raise rep_err

            # Re-validate repaired report
            logger.info("Validating repaired report grounding and citations...")
            try:
                self.grounding_validator.validate(repaired_report, sanitized_results)
                report = repaired_report
                report_repair_succeeded = True
                logger.info("Report repair succeeded.")
            except GroundingValidationError as repair_exc:
                repair_exc.analysis_plan = plan
                repair_exc.analysis_results = sanitized_results
                repair_exc.report = repaired_report
                repair_exc.failure_stage = "grounding_validation"
                repair_exc.report_repair_attempted = report_repair_attempted
                repair_exc.report_repair_succeeded = report_repair_succeeded
                raise repair_exc

        logger.info("Analysis completed successfully.")

        # 9. Assemble and return trusted response
        return AnalysisResponse(
            question=question,
            dataset_summary=summary,
            analysis_plan=plan,
            analysis_results=sanitized_results,
            findings=report.findings,
            limitations=report.limitations,
            recommendations=report.recommendations,
            plan_repair_attempted=self.plan_repair_attempted,
            plan_repair_succeeded=self.plan_repair_succeeded,
            report_repair_attempted=report_repair_attempted,
            report_repair_succeeded=report_repair_succeeded
        )
