import json
from google import genai
from app.config import settings
from app.providers.base import BaseProvider
from app.models.responses import DatasetSummary
from app.models.analysis import AnalysisPlan, ProviderReport, AnalysisResult
from app.core.exceptions import ProviderError
from app.prompts.planner import PLANNER_SYSTEM_PROMPT, format_planner_user_prompt, REPAIR_SYSTEM_PROMPT, format_repair_user_prompt
from app.prompts.reporter import REPORTER_SYSTEM_PROMPT, format_reporter_user_prompt

class GeminiProvider(BaseProvider):
    def _get_client(self) -> genai.Client:
        if not settings.GEMINI_API_KEY or not settings.GEMINI_API_KEY.strip():
            raise ProviderError("Gemini API Key is not set in settings.", status_code=400)
        return genai.Client(api_key=settings.GEMINI_API_KEY)

    async def create_analysis_plan(self, question: str, summary: DatasetSummary) -> AnalysisPlan:
        """Invokes Gemini model using GenAI Interactions API to create an AnalysisPlan.

        Performs explicit Pydantic validation on the JSON output.
        """
        client = self._get_client()
        sys_prompt = PLANNER_SYSTEM_PROMPT.format(max_steps=settings.MAX_ANALYSIS_STEPS)
        user_prompt = format_planner_user_prompt(question, summary.model_dump_json(indent=2))
        input_text = f"System Instructions:\n{sys_prompt}\n\nUser Input:\n{user_prompt}"

        try:
            interaction = client.interactions.create(
                model=settings.GEMINI_MODEL,
                input=input_text,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": AnalysisPlan.model_json_schema(),
                },
            )

            # Access the output text directly using output_text as required by the Interactions contract
            text_output = getattr(interaction, "output_text", None)
            if not text_output:
                raise ProviderError("Gemini returned no structured output.")

            return AnalysisPlan.model_validate_json(text_output)

        except Exception as e:
            error_msg = str(e)
            if settings.GEMINI_API_KEY in error_msg:
                error_msg = error_msg.replace(settings.GEMINI_API_KEY, "GEMINI_API_KEY")
            raise ProviderError(f"Gemini planner request failed: {error_msg}")

    async def repair_analysis_plan(
        self,
        question: str,
        dataset_summary: DatasetSummary,
        invalid_plan: AnalysisPlan,
        validation_feedback: str,
    ) -> AnalysisPlan:
        """Invokes Gemini model using GenAI Interactions API to repair an AnalysisPlan."""
        client = self._get_client()
        sys_prompt = REPAIR_SYSTEM_PROMPT.format(max_steps=settings.MAX_ANALYSIS_STEPS)
        user_prompt = format_repair_user_prompt(
            question,
            dataset_summary.model_dump_json(indent=2),
            invalid_plan.model_dump_json(indent=2),
            validation_feedback
        )
        input_text = f"System Instructions:\n{sys_prompt}\n\nUser Input:\n{user_prompt}"

        try:
            interaction = client.interactions.create(
                model=settings.GEMINI_MODEL,
                input=input_text,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": AnalysisPlan.model_json_schema(),
                },
            )

            text_output = getattr(interaction, "output_text", None)
            if not text_output:
                raise ProviderError("Gemini returned no structured output during repair.")

            return AnalysisPlan.model_validate_json(text_output)

        except Exception as e:
            error_msg = str(e)
            if settings.GEMINI_API_KEY in error_msg:
                error_msg = error_msg.replace(settings.GEMINI_API_KEY, "GEMINI_API_KEY")
            raise ProviderError(f"Gemini planner repair request failed: {error_msg}")

    async def generate_report(
        self, question: str, summary: DatasetSummary, results: list[AnalysisResult]
    ) -> ProviderReport:
        """Invokes Gemini model using GenAI Interactions API to generate a ProviderReport.

        Performs explicit Pydantic validation on the JSON output.
        """
        client = self._get_client()
        sys_prompt = REPORTER_SYSTEM_PROMPT

        results_serialized = json.dumps([r.model_dump() for r in results], indent=2)
        user_prompt = format_reporter_user_prompt(
            question,
            summary.model_dump_json(indent=2),
            results_serialized
        )
        input_text = f"System Instructions:\n{sys_prompt}\n\nUser Input:\n{user_prompt}"

        try:
            interaction = client.interactions.create(
                model=settings.GEMINI_MODEL,
                input=input_text,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": ProviderReport.model_json_schema(),
                },
            )

            text_output = getattr(interaction, "output_text", None)
            if not text_output:
                raise ProviderError("Gemini returned no structured output.")

            return ProviderReport.model_validate_json(text_output)

        except Exception as e:
            error_msg = str(e)
            if settings.GEMINI_API_KEY in error_msg:
                error_msg = error_msg.replace(settings.GEMINI_API_KEY, "GEMINI_API_KEY")
            raise ProviderError(f"Gemini reporter request failed: {error_msg}")
