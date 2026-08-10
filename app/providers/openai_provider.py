import json
from openai import OpenAI
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

class OpenAIProvider(BaseProvider):
    def _get_client(self) -> OpenAI:
        if not settings.OPENAI_API_KEY or not settings.OPENAI_API_KEY.strip():
            raise ProviderError("OpenAI API Key is not set in settings.", status_code=400)
        return OpenAI(api_key=settings.OPENAI_API_KEY)

    async def create_analysis_plan(self, question: str, summary: DatasetSummary) -> AnalysisPlan:
        """Invokes OpenAI completions API with Responses API structured parsing for the AnalysisPlan."""
        client = self._get_client()
        sys_prompt = PLANNER_SYSTEM_PROMPT.format(max_steps=settings.MAX_ANALYSIS_STEPS)
        user_prompt = format_planner_user_prompt(question, summary.model_dump_json(indent=2))

        try:
            response = client.responses.parse(
                model=settings.OPENAI_MODEL,
                input=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                text_format=AnalysisPlan
            )

            parsed_plan = getattr(response, "output_parsed", None)
            if not parsed_plan:
                raise ProviderError("OpenAI provider returned empty parsed response (refused or empty).")
            return parsed_plan

        except Exception as e:
            error_msg = str(e)
            if settings.OPENAI_API_KEY in error_msg:
                error_msg = error_msg.replace(settings.OPENAI_API_KEY, "OPENAI_API_KEY")
            raise ProviderError(f"OpenAI planner request failed: {error_msg}")

    async def repair_analysis_plan(
        self,
        question: str,
        dataset_summary: DatasetSummary,
        invalid_plan: AnalysisPlan,
        validation_feedback: str,
    ) -> AnalysisPlan:
        """Invokes OpenAI completions API with Responses API structured parsing to repair an AnalysisPlan."""
        client = self._get_client()
        sys_prompt = REPAIR_SYSTEM_PROMPT.format(max_steps=settings.MAX_ANALYSIS_STEPS)
        user_prompt = format_repair_user_prompt(
            question,
            dataset_summary.model_dump_json(indent=2),
            invalid_plan.model_dump_json(indent=2),
            validation_feedback
        )

        try:
            response = client.responses.parse(
                model=settings.OPENAI_MODEL,
                input=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                text_format=AnalysisPlan
            )

            parsed_plan = getattr(response, "output_parsed", None)
            if not parsed_plan:
                raise ProviderError("OpenAI provider returned empty parsed response during repair (refused or empty).")
            return parsed_plan

        except Exception as e:
            error_msg = str(e)
            if settings.OPENAI_API_KEY in error_msg:
                error_msg = error_msg.replace(settings.OPENAI_API_KEY, "OPENAI_API_KEY")
            raise ProviderError(f"OpenAI planner repair request failed: {error_msg}")

    async def generate_report(
        self, question: str, summary: DatasetSummary, results: list[AnalysisResult]
    ) -> ProviderReport:
        """Invokes OpenAI completions API with Responses API structured parsing for the ProviderReport."""
        client = self._get_client()
        sys_prompt = REPORTER_SYSTEM_PROMPT

        results_serialized = json.dumps([res.model_dump() for res in results], indent=2)
        user_prompt = format_reporter_user_prompt(
            question,
            summary.model_dump_json(indent=2),
            results_serialized
        )

        try:
            response = client.responses.parse(
                model=settings.OPENAI_MODEL,
                input=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                text_format=ProviderReport
            )

            parsed_report = getattr(response, "output_parsed", None)
            if not parsed_report:
                raise ProviderError("OpenAI provider returned empty parsed report (refused or empty).")
            return parsed_report

        except Exception as e:
            error_msg = str(e)
            if settings.OPENAI_API_KEY in error_msg:
                error_msg = error_msg.replace(settings.OPENAI_API_KEY, "OPENAI_API_KEY")
            raise ProviderError(f"OpenAI reporter request failed: {error_msg}")

    async def repair_report(
        self,
        question: str,
        dataset_summary: DatasetSummary,
        analysis_results: list[AnalysisResult],
        invalid_report: ProviderReport,
        validation_feedback: str,
    ) -> ProviderReport:
        """Invokes OpenAI completions API with Responses API structured parsing to repair ProviderReport."""
        client = self._get_client()
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
            response = client.responses.parse(
                model=settings.OPENAI_MODEL,
                input=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                text_format=ProviderReport
            )

            parsed_report = getattr(response, "output_parsed", None)
            if not parsed_report:
                raise ProviderError("OpenAI provider returned empty parsed repaired report.")
            return parsed_report

        except Exception as e:
            error_msg = str(e)
            if settings.OPENAI_API_KEY in error_msg:
                error_msg = error_msg.replace(settings.OPENAI_API_KEY, "OPENAI_API_KEY")
            raise ProviderError(f"OpenAI report repair request failed: {error_msg}")
