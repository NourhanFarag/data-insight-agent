import pytest
import asyncio
from unittest.mock import MagicMock, patch
from app.config import settings
from app.providers import get_provider
from app.providers.mock_provider import MockProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.ollama_provider import OllamaProvider
from app.core.exceptions import ProviderError
from app.models.responses import DatasetSummary
from app.models.analysis import (
    AnalysisPlan,
    AnalysisStep,
    ProviderReport,
    AnalysisOperation,
    ConfidenceLevel,
    Finding,
    Recommendation,
    RecommendationPriority,
    AnalysisResult
)

# 1. Provider Factory Tests

def test_provider_factory_mock():
    settings.AI_PROVIDER = "mock"
    prov = get_provider()
    assert isinstance(prov, MockProvider)

def test_provider_factory_invalid():
    settings.AI_PROVIDER = "invalid_prov"
    with pytest.raises(ProviderError) as exc:
        get_provider()
    assert "Unsupported AI provider" in str(exc.value)

def test_provider_factory_gemini_missing_key():
    settings.AI_PROVIDER = "gemini"
    settings.GEMINI_API_KEY = ""
    with pytest.raises(ProviderError) as exc:
        get_provider()
    assert "GEMINI_API_KEY is not configured" in str(exc.value)

def test_provider_factory_openai_missing_key():
    settings.AI_PROVIDER = "openai"
    settings.OPENAI_API_KEY = ""
    with pytest.raises(ProviderError) as exc:
        get_provider()
    assert "OPENAI_API_KEY is not configured" in str(exc.value)


# 2. Gemini Provider Unit Tests

@pytest.fixture
def dummy_summary():
    return DatasetSummary(
        row_count=100, column_count=2, column_names=["revenue", "department"],
        inferred_data_types={"revenue": "numeric", "department": "categorical"},
        missing_value_count={"revenue": 0, "department": 0},
        numeric_columns=["revenue"], categorical_columns=["department"]
    )

@patch("app.providers.gemini_provider.genai.Client")
def test_gemini_provider_plan_success(mock_client_class, dummy_summary):
    # Test 1: Valid output_text -> valid AnalysisPlan
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    # Uses output_text as required
    mock_response.output_text = '{"objective": "Test plan", "steps": [{"step_id": "step_1", "operation": "COUNT"}]}'
    mock_client.interactions.create.return_value = mock_response

    settings.GEMINI_API_KEY = "test_gemini_key"
    settings.GEMINI_MODEL = "gemini-3.6-flash"

    provider = GeminiProvider()
    plan = asyncio.run(provider.create_analysis_plan("What is total row count?", dummy_summary))

    assert plan.objective == "Test plan"
    assert len(plan.steps) == 1
    assert plan.steps[0].operation == AnalysisOperation.COUNT

    mock_client.interactions.create.assert_called_once()

@patch("app.providers.gemini_provider.genai.Client")
def test_gemini_provider_report_success(mock_client_class, dummy_summary):
    # Test 2: Valid output_text -> valid ProviderReport
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.output_text = (
        '{"findings": [{'
        '  "id": "finding_1", "title": "Scale", "explanation": "Count is 100", '
        '  "evidence_refs": ["result_1"], "confidence": "HIGH"'
        '}],'
        '"limitations": ["data limits"],'
        '"recommendations": [{'
        '  "id": "recommendation_1", "priority": "HIGH", "action": "Do X", '
        '  "rationale": "Reason", "finding_refs": ["finding_1"]'
        '}]}'
    )
    mock_client.interactions.create.return_value = mock_response

    settings.GEMINI_API_KEY = "test_gemini_key"
    settings.GEMINI_MODEL = "gemini-3.6-flash"

    provider = GeminiProvider()
    report = asyncio.run(provider.generate_report("What is total row count?", dummy_summary, []))

    assert len(report.findings) == 1
    assert report.findings[0].id == "finding_1"
    assert report.limitations == ["data limits"]
    assert report.recommendations[0].priority == RecommendationPriority.HIGH

@patch("app.providers.gemini_provider.genai.Client")
def test_gemini_provider_output_text_none(mock_client_class, dummy_summary):
    # Test 3: output_text = None -> ProviderError
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.output_text = None
    mock_client.interactions.create.return_value = mock_response

    settings.GEMINI_API_KEY = "test_gemini_key"
    provider = GeminiProvider()

    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.create_analysis_plan("Test", dummy_summary))
    assert "Gemini returned no structured output" in str(exc.value)

@patch("app.providers.gemini_provider.genai.Client")
def test_gemini_provider_output_text_empty(mock_client_class, dummy_summary):
    # Test 4: output_text = "" -> ProviderError
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.output_text = ""
    mock_client.interactions.create.return_value = mock_response

    settings.GEMINI_API_KEY = "test_gemini_key"
    provider = GeminiProvider()

    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.create_analysis_plan("Test", dummy_summary))
    assert "Gemini returned no structured output" in str(exc.value)

@patch("app.providers.gemini_provider.genai.Client")
def test_gemini_provider_malformed_json(mock_client_class, dummy_summary):
    # Test 5: malformed JSON -> ProviderError
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.output_text = '{"malformed_json: true'
    mock_client.interactions.create.return_value = mock_response

    settings.GEMINI_API_KEY = "test_gemini_key"
    provider = GeminiProvider()

    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.create_analysis_plan("Test", dummy_summary))
    assert "Gemini planner request failed" in str(exc.value)

@patch("app.providers.gemini_provider.genai.Client")
def test_gemini_provider_schema_invalid(mock_client_class, dummy_summary):
    # Test 6: schema-invalid JSON -> ProviderError
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    # Missing required field 'steps'
    mock_response.output_text = '{"objective": "Incomplete plan without steps"}'
    mock_client.interactions.create.return_value = mock_response

    settings.GEMINI_API_KEY = "test_gemini_key"
    provider = GeminiProvider()

    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.create_analysis_plan("Test", dummy_summary))
    assert "validation error" in str(exc.value).lower()

@patch("app.providers.gemini_provider.genai.Client")
def test_gemini_provider_error_handling(mock_client_class, dummy_summary):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.interactions.create.side_effect = Exception("API failed with key test_gemini_key_secret")

    settings.GEMINI_API_KEY = "test_gemini_key_secret"
    provider = GeminiProvider()

    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.create_analysis_plan("What is the count?", dummy_summary))

    assert "test_gemini_key_secret" not in str(exc.value)
    assert "GEMINI_API_KEY" in str(exc.value)


# 3. OpenAI Provider Unit Tests

@patch("app.providers.openai_provider.OpenAI")
def test_openai_provider_plan_success(mock_openai_class, dummy_summary):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_plan = AnalysisPlan(
        objective="Test objective",
        steps=[
            AnalysisStep(step_id="step_1", operation=AnalysisOperation.COUNT)
        ]
    )

    mock_response_object = MagicMock()
    mock_response_object.output_parsed = mock_plan
    mock_client.responses.parse.return_value = mock_response_object

    settings.OPENAI_API_KEY = "test_openai_key"
    settings.OPENAI_MODEL = "gpt-5.6-luna"

    provider = OpenAIProvider()
    plan = asyncio.run(provider.create_analysis_plan("Analyze rows", dummy_summary))

    assert plan.objective == "Test objective"
    assert len(plan.steps) == 1
    assert plan.steps[0].step_id == "step_1"

    mock_client.responses.parse.assert_called_once()

@patch("app.providers.openai_provider.OpenAI")
def test_openai_provider_report_success(mock_openai_class, dummy_summary):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_report = ProviderReport(
        findings=[
            Finding(id="finding_1", title="Scale", explanation="Count is 100", evidence_refs=["result_1"], confidence=ConfidenceLevel.HIGH)
        ],
        limitations=["mock limitations"],
        recommendations=[
            Recommendation(id="recommendation_1", priority=RecommendationPriority.MEDIUM, action="A", rationale="R", finding_refs=["finding_1"])
        ]
    )

    mock_response_object = MagicMock()
    mock_response_object.output_parsed = mock_report
    mock_client.responses.parse.return_value = mock_response_object

    settings.OPENAI_API_KEY = "test_openai_key"
    settings.OPENAI_MODEL = "gpt-5.6-luna"

    provider = OpenAIProvider()
    report = asyncio.run(provider.generate_report("Count rows", dummy_summary, []))

    assert len(report.findings) == 1
    assert report.findings[0].id == "finding_1"
    assert report.limitations == ["mock limitations"]

    mock_client.responses.parse.assert_called_once()

@patch("app.providers.openai_provider.OpenAI")
def test_openai_provider_error_handling(mock_openai_class, dummy_summary):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.responses.parse.side_effect = Exception("OpenAI failed with key test_openai_key_secret")

    settings.OPENAI_API_KEY = "test_openai_key_secret"
    provider = OpenAIProvider()

    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.generate_report("Count rows", dummy_summary, []))

    assert "test_openai_key_secret" not in str(exc.value)
    assert "OPENAI_API_KEY" in str(exc.value)

@patch("app.providers.openai_provider.OpenAI")
def test_openai_provider_refusal_empty_output(mock_openai_class, dummy_summary):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response_object = MagicMock()
    mock_response_object.output_parsed = None
    mock_client.responses.parse.return_value = mock_response_object

    settings.OPENAI_API_KEY = "test_openai_key"
    provider = OpenAIProvider()

    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.create_analysis_plan("Test", dummy_summary))
    assert "empty parsed response" in str(exc.value)


# 4. Ollama Provider Unit Tests

@patch("app.providers.ollama_provider.Client")
def test_ollama_provider_plan_success(mock_client_class, dummy_summary):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.message.content = '{"objective": "Test plan", "steps": [{"step_id": "step_1", "operation": "COUNT"}]}'
    mock_client.chat.return_value = mock_response

    settings.OLLAMA_BASE_URL = "http://localhost:11434"
    settings.OLLAMA_MODEL = "qwen3:8b"
    settings.OLLAMA_TEMPERATURE = 0.0

    provider = OllamaProvider()
    plan = asyncio.run(provider.create_analysis_plan("What is total row count?", dummy_summary))

    assert plan.objective == "Test plan"
    assert len(plan.steps) == 1
    assert plan.steps[0].operation == AnalysisOperation.COUNT

    mock_client.chat.assert_called_once()
    args, kwargs = mock_client.chat.call_args
    assert kwargs["model"] == "qwen3:8b"
    assert kwargs["options"]["temperature"] == 0.0

@patch("app.providers.ollama_provider.Client")
def test_ollama_provider_report_success(mock_client_class, dummy_summary):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.message.content = (
        '{"findings": [{'
        '  "id": "finding_1", "title": "Scale", "explanation": "Count is 100", '
        '  "evidence_refs": ["result_1"], "confidence": "HIGH"'
        '}],'
        '"limitations": ["data limits"],'
        '"recommendations": [{'
        '  "id": "recommendation_1", "priority": "HIGH", "action": "Do X", '
        '  "rationale": "Reason", "finding_refs": ["finding_1"]'
        '}]}'
    )
    mock_client.chat.return_value = mock_response

    provider = OllamaProvider()
    report = asyncio.run(provider.generate_report("What is total row count?", dummy_summary, []))

    assert len(report.findings) == 1
    assert report.findings[0].id == "finding_1"
    assert report.limitations == ["data limits"]

@patch("app.providers.ollama_provider.Client")
def test_ollama_provider_malformed_json(mock_client_class, dummy_summary):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.message.content = '{"malformed_json: true'
    mock_client.chat.return_value = mock_response

    provider = OllamaProvider()
    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.create_analysis_plan("Test", dummy_summary))
    assert "Ollama planner request failed" in str(exc.value)

@patch("app.providers.ollama_provider.Client")
def test_ollama_provider_schema_invalid(mock_client_class, dummy_summary):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.message.content = '{"objective": "Missing steps required field"}'
    mock_client.chat.return_value = mock_response

    provider = OllamaProvider()
    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.create_analysis_plan("Test", dummy_summary))
    assert "validation error" in str(exc.value).lower()

@patch("app.providers.ollama_provider.Client")
def test_ollama_provider_blank_output(mock_client_class, dummy_summary):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.message.content = "   "
    mock_client.chat.return_value = mock_response

    provider = OllamaProvider()
    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.create_analysis_plan("Test", dummy_summary))
    assert "no structured output" in str(exc.value)

@patch("app.providers.ollama_provider.Client")
def test_ollama_provider_none_output(mock_client_class, dummy_summary):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.message.content = None
    mock_client.chat.return_value = mock_response

    provider = OllamaProvider()
    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.create_analysis_plan("Test", dummy_summary))
    assert "no structured output" in str(exc.value)

@patch("app.providers.ollama_provider.Client")
def test_ollama_provider_connection_failure(mock_client_class, dummy_summary):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.chat.side_effect = ConnectionError("Could not connect to http://localhost:11434")

    provider = OllamaProvider()
    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.create_analysis_plan("Test", dummy_summary))
    assert "Ollama is not available" in str(exc.value)

@patch("app.providers.ollama_provider.Client")
def test_ollama_provider_payload_boundary(mock_client_class):
    """Verifies that the planner input to Ollama contains question & summary, but excludes raw CSV secrets."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.message.content = '{"objective": "Test", "steps": []}'
    mock_client.chat.return_value = mock_response

    summary = DatasetSummary(
        row_count=1, column_count=1, column_names=["col1"],
        inferred_data_types={"col1": "categorical"},
        missing_value_count={"col1": 0},
        numeric_columns=[], categorical_columns=["col1"]
    )

    provider = OllamaProvider()
    asyncio.run(provider.create_analysis_plan("Which segment?", summary))

    mock_client.chat.assert_called_once()
    args, kwargs = mock_client.chat.call_args

    messages = kwargs["messages"]
    sys_content = messages[0]["content"]
    user_content = messages[1]["content"]

    raw_secret_marker = "FLAG_CSV_RAW_CELL_SECRET"
    assert raw_secret_marker not in sys_content
    assert raw_secret_marker not in user_content
    assert "Which segment?" in user_content
    assert "col1" in user_content


# 5. Ollama Schema Compatibility and Sanitization Tests

def test_ollama_schema_compatibility_regression():
    """Verify that Ollama compatible models do not contain 'pattern' constraints, while canonical models still do."""
    from app.providers.ollama_schemas import OllamaAnalysisPlan, OllamaProviderReport
    from app.models.analysis import AnalysisPlan, ProviderReport

    def has_key_recursive(d: dict, key: str) -> bool:
        if key in d:
            return True
        for v in d.values():
            if isinstance(v, dict):
                if has_key_recursive(v, key):
                    return True
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        if has_key_recursive(item, key):
                            return True
        return False

    # Ollama compatibility models must have NO pattern constraints
    assert not has_key_recursive(OllamaAnalysisPlan.model_json_schema(), "pattern")
    assert not has_key_recursive(OllamaProviderReport.model_json_schema(), "pattern")

    # Canonical domain models MUST have pattern constraints for safety validations
    assert has_key_recursive(AnalysisPlan.model_json_schema(), "pattern")
    assert has_key_recursive(ProviderReport.model_json_schema(), "pattern")


def test_ollama_to_canonical_model_conversion_success():
    """Verify that valid schema structures from Ollama compatibility models successfully convert to canonical models."""
    from app.providers.ollama_schemas import OllamaAnalysisPlan
    from app.models.analysis import AnalysisPlan

    raw_plan_data = {
        "objective": "Identify high value clients",
        "steps": [
            {
                "step_id": "step_1",
                "operation": "TOP_VALUES",
                "column": "revenue",
                "limit": 10,
                "reason": "Identify top revenue generating customers"
            }
        ]
    }
    # Ollama schema accepts this
    ollama_plan = OllamaAnalysisPlan.model_validate(raw_plan_data)

    # Canonical schema accepts and validates successfully
    canonical_plan = AnalysisPlan.model_validate(ollama_plan.model_dump())
    assert canonical_plan.objective == "Identify high value clients"
    assert canonical_plan.steps[0].step_id == "step_1"


def test_ollama_to_canonical_step_id_failure():
    """Verify that an invalid step ID pattern (e.g. step_0) parses in Ollama schema but fails canonical conversion."""
    from app.providers.ollama_schemas import OllamaAnalysisPlan
    from app.models.analysis import AnalysisPlan
    from pydantic import ValidationError

    raw_plan_data = {
        "objective": "Identify high value clients",
        "steps": [
            {
                "step_id": "step_0", # Invalid step id (canonical matches step_[1-9]\d*)
                "operation": "COUNT"
            }
        ]
    }
    # Ollama schema should accept it without issue
    ollama_plan = OllamaAnalysisPlan.model_validate(raw_plan_data)

    # Canonical conversion must fail validation
    with pytest.raises(ValidationError) as exc:
        AnalysisPlan.model_validate(ollama_plan.model_dump())
    assert "step_id" in str(exc.value)


def test_ollama_to_canonical_report_id_failures():
    """Verify that invalid finding or recommendation ID patterns parse in Ollama schema but fail canonical conversion."""
    from app.providers.ollama_schemas import OllamaProviderReport
    from app.models.analysis import ProviderReport
    from pydantic import ValidationError

    # Test Finding ID failure (e.g. finding_0)
    raw_report_invalid_finding = {
        "findings": [
            {
                "id": "finding_0", # Invalid (pattern matches finding_[1-9]\d*)
                "title": "Finding Title",
                "explanation": "Explanation",
                "evidence_refs": ["result_1"],
                "confidence": "HIGH"
            }
        ],
        "limitations": [],
        "recommendations": []
    }
    ollama_report = OllamaProviderReport.model_validate(raw_report_invalid_finding)
    with pytest.raises(ValidationError) as exc:
        ProviderReport.model_validate(ollama_report.model_dump())
    assert "findings" in str(exc.value)

    # Test Recommendation ID failure (e.g. recommendation_0)
    raw_report_invalid_rec = {
        "findings": [
            {
                "id": "finding_1",
                "title": "Finding Title",
                "explanation": "Explanation",
                "evidence_refs": ["result_1"],
                "confidence": "HIGH"
            }
        ],
        "limitations": [],
        "recommendations": [
            {
                "id": "recommendation_0", # Invalid (pattern matches recommendation_[1-9]\d*)
                "priority": "HIGH",
                "action": "Action",
                "rationale": "Rationale",
                "finding_refs": ["finding_1"]
            }
        ]
    }
    ollama_report_rec = OllamaProviderReport.model_validate(raw_report_invalid_rec)
    with pytest.raises(ValidationError) as exc:
        ProviderReport.model_validate(ollama_report_rec.model_dump())
    assert "recommendations" in str(exc.value)


@patch("app.providers.ollama_provider.Client")
def test_ollama_provider_grammar_error_sanitization(mock_client_class, dummy_summary):
    """Verify that Ollama 400 grammar parsing errors are caught and sanitized as a clear ProviderError."""
    from ollama import ResponseError

    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # 400 ResponseError indicating grammar parsing error
    re = ResponseError("Failed to initialize samplers: failed to parse grammar", 400)
    mock_client.chat.side_effect = re

    provider = OllamaProvider()
    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.create_analysis_plan("Test Question?", dummy_summary))

    assert "Schema grammar compatibility issue or invalid request" in str(exc.value)
    # Ensure internal exception details or prompt internals are not exposed in the error message
    assert "Failed to initialize samplers" not in str(exc.value)
    assert "failed to parse grammar" not in str(exc.value)
