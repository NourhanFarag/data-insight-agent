import json
import logging
from ollama import Client, ResponseError
from app.config import settings
from app.providers.base import BaseProvider
from app.models.responses import DatasetSummary
from app.models.analysis import AnalysisPlan, ProviderReport, AnalysisResult
from app.core.exceptions import ProviderError
from app.prompts.planner import PLANNER_SYSTEM_PROMPT, format_planner_user_prompt, REPAIR_SYSTEM_PROMPT, format_repair_user_prompt
from app.prompts.reporter import (
    REPORTER_SYSTEM_PROMPT,
    format_reporter_user_prompt,
    REPORT_REPAIR_SYSTEM_PROMPT,
    format_report_repair_user_prompt,
)
from app.providers.ollama_schemas import OllamaAnalysisPlan, OllamaProviderReport

logger = logging.getLogger(__name__)

class OllamaProvider(BaseProvider):
    def _get_client(self) -> Client:
        """Returns a configured local Ollama client."""
        # Provider creation should not require the service to be running.
        return Client(host=settings.OLLAMA_BASE_URL)

    async def create_analysis_plan(self, question: str, summary: DatasetSummary) -> AnalysisPlan:
        """Invokes local Ollama model to generate a structured AnalysisPlan."""
        sys_prompt = PLANNER_SYSTEM_PROMPT.format(max_steps=settings.MAX_ANALYSIS_STEPS)
        user_prompt = format_planner_user_prompt(question, summary.model_dump_json(indent=2))

        try:
            client = self._get_client()
            response = client.chat(
                model=settings.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format=OllamaAnalysisPlan.model_json_schema(),
                options={
                    "temperature": settings.OLLAMA_TEMPERATURE
                }
            )

            content = getattr(response.message, "content", None)
            if not content or not content.strip():
                raise ProviderError("Ollama returned no structured output.")

            try:
                raw_plan = OllamaAnalysisPlan.model_validate_json(content)
            except Exception as e:
                raise ProviderError(f"Ollama plan parsing failed: {e}")

            try:
                plan = AnalysisPlan.model_validate(raw_plan.model_dump())
                return plan
            except Exception as e:
                raise ProviderError(f"Ollama plan schema validation failed: {e}")

        except ConnectionError as ce:
            raise ProviderError(
                "Ollama is not available at the configured local URL. Start Ollama and ensure the configured model is installed."
            )
        except ResponseError as re:
            logger.error("Ollama ResponseError: status_code=%s, error=%s", re.status_code, re.error)
            if re.status_code == 400 or "grammar" in str(re).lower() or "sampler" in str(re).lower():
                raise ProviderError(
                    "Ollama request failed: Schema grammar compatibility issue or invalid request."
                )
            raise ProviderError(f"Ollama planner request failed: {re}")
        except Exception as e:
            error_msg = str(e)
            if "connect" in error_msg.lower() or "connection" in error_msg.lower():
                raise ProviderError(
                    "Ollama is not available at the configured local URL. Start Ollama and ensure the configured model is installed."
                )
            raise ProviderError(f"Ollama planner request failed: {error_msg}")

    async def repair_analysis_plan(
        self,
        question: str,
        dataset_summary: DatasetSummary,
        invalid_plan: AnalysisPlan,
        validation_feedback: str,
    ) -> AnalysisPlan:
        """Invokes local Ollama model to repair an invalid AnalysisPlan."""
        sys_prompt = REPAIR_SYSTEM_PROMPT.format(max_steps=settings.MAX_ANALYSIS_STEPS)
        user_prompt = format_repair_user_prompt(
            question,
            dataset_summary.model_dump_json(indent=2),
            invalid_plan.model_dump_json(indent=2),
            validation_feedback
        )

        try:
            client = self._get_client()
            response = client.chat(
                model=settings.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format=OllamaAnalysisPlan.model_json_schema(),
                options={
                    "temperature": settings.OLLAMA_TEMPERATURE
                }
            )

            content = getattr(response.message, "content", None)
            if not content or not content.strip():
                raise ProviderError("Ollama returned no structured output during repair.")

            try:
                raw_plan = OllamaAnalysisPlan.model_validate_json(content)
            except Exception as e:
                raise ProviderError(f"Ollama repaired plan parsing failed: {e}")

            try:
                plan = AnalysisPlan.model_validate(raw_plan.model_dump())
                return plan
            except Exception as e:
                raise ProviderError(f"Ollama repaired plan schema validation failed: {e}")

        except ConnectionError as ce:
            raise ProviderError(
                "Ollama is not available at the configured local URL. Start Ollama and ensure the configured model is installed."
            )
        except ResponseError as re:
            logger.error("Ollama ResponseError during repair: status_code=%s, error=%s", re.status_code, re.error)
            if re.status_code == 400 or "grammar" in str(re).lower() or "sampler" in str(re).lower():
                raise ProviderError(
                    "Ollama request failed: Schema grammar compatibility issue or invalid request."
                )
            raise ProviderError(f"Ollama plan repair request failed: {re}")
        except Exception as e:
            error_msg = str(e)
            if "connect" in error_msg.lower() or "connection" in error_msg.lower():
                raise ProviderError(
                    "Ollama is not available at the configured local URL. Start Ollama and ensure the configured model is installed."
                )
            raise ProviderError(f"Ollama plan repair request failed: {error_msg}")

    async def generate_report(
        self, question: str, summary: DatasetSummary, results: list[AnalysisResult]
    ) -> ProviderReport:
        """Invokes local Ollama model to generate a structured ProviderReport."""
        sys_prompt = REPORTER_SYSTEM_PROMPT

        results_serialized = json.dumps([res.model_dump() for res in results], indent=2)
        user_prompt = format_reporter_user_prompt(
            question,
            summary.model_dump_json(indent=2),
            results_serialized
        )

        try:
            client = self._get_client()
            response = client.chat(
                model=settings.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format=OllamaProviderReport.model_json_schema(),
                options={
                    "temperature": settings.OLLAMA_TEMPERATURE
                }
            )

            content = getattr(response.message, "content", None)
            if not content or not content.strip():
                raise ProviderError("Ollama returned no structured output.")

            try:
                raw_report = OllamaProviderReport.model_validate_json(content)
            except Exception as e:
                raise ProviderError(f"Ollama report parsing failed: {e}")

            try:
                report = ProviderReport.model_validate(raw_report.model_dump())
                return report
            except Exception as e:
                raise ProviderError(f"Ollama report schema validation failed: {e}")

        except ConnectionError as ce:
            raise ProviderError(
                "Ollama is not available at the configured local URL. Start Ollama and ensure the configured model is installed."
            )
        except ResponseError as re:
            logger.error("Ollama ResponseError: status_code=%s, error=%s", re.status_code, re.error)
            if re.status_code == 400 or "grammar" in str(re).lower() or "sampler" in str(re).lower():
                raise ProviderError(
                    "Ollama request failed: Schema grammar compatibility issue or invalid request."
                )
            raise ProviderError(f"Ollama reporter request failed: {re}")
        except Exception as e:
            error_msg = str(e)
            if "connect" in error_msg.lower() or "connection" in error_msg.lower():
                raise ProviderError(
                    "Ollama is not available at the configured local URL. Start Ollama and ensure the configured model is installed."
                )
            raise ProviderError(f"Ollama reporter request failed: {error_msg}")

    async def repair_report(
        self,
        question: str,
        dataset_summary: DatasetSummary,
        analysis_results: list[AnalysisResult],
        invalid_report: ProviderReport,
        validation_feedback: str,
    ) -> ProviderReport:
        """Invokes local Ollama model to repair an invalid ProviderReport."""
        sys_prompt = REPORT_REPAIR_SYSTEM_PROMPT

        results_serialized = json.dumps([res.model_dump() for res in analysis_results], indent=2)
        user_prompt = format_report_repair_user_prompt(
            question,
            dataset_summary.model_dump_json(indent=2),
            results_serialized,
            invalid_report.model_dump_json(indent=2),
            validation_feedback
        )

        try:
            client = self._get_client()
            response = client.chat(
                model=settings.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format=OllamaProviderReport.model_json_schema(),
                options={
                    "temperature": settings.OLLAMA_TEMPERATURE
                }
            )

            content = getattr(response.message, "content", None)
            if not content or not content.strip():
                raise ProviderError("Ollama returned no structured output during report repair.")

            try:
                raw_report = OllamaProviderReport.model_validate_json(content)
            except Exception as e:
                raise ProviderError(f"Ollama repaired report parsing failed: {e}")

            try:
                report = ProviderReport.model_validate(raw_report.model_dump())
                return report
            except Exception as e:
                raise ProviderError(f"Ollama repaired report schema validation failed: {e}")

        except ConnectionError as ce:
            raise ProviderError(
                "Ollama is not available at the configured local URL. Start Ollama and ensure the configured model is installed."
            )
        except ResponseError as re:
            logger.error("Ollama ResponseError during report repair: status_code=%s, error=%s", re.status_code, re.error)
            if re.status_code == 400 or "grammar" in str(re).lower() or "sampler" in str(re).lower():
                raise ProviderError(
                    "Ollama request failed: Schema grammar compatibility issue or invalid request."
                )
            raise ProviderError(f"Ollama report repair request failed: {re}")
        except Exception as e:
            error_msg = str(e)
            if "connect" in error_msg.lower() or "connection" in error_msg.lower():
                raise ProviderError(
                    "Ollama is not available at the configured local URL. Start Ollama and ensure the configured model is installed."
                )
            raise ProviderError(f"Ollama report repair request failed: {error_msg}")
