from app.config import settings
from app.providers.base import BaseProvider
from app.providers.mock_provider import MockProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.ollama_provider import OllamaProvider
from app.core.exceptions import ProviderError

def get_provider() -> BaseProvider:
    """Factory to retrieve a configured BaseProvider.

    Verifies that the required configuration options are present.
    """
    provider_name = settings.AI_PROVIDER.lower()

    if provider_name == "mock":
        return MockProvider()

    elif provider_name == "ollama":
        return OllamaProvider()

    elif provider_name == "gemini":
        if not settings.GEMINI_API_KEY or not settings.GEMINI_API_KEY.strip():
            raise ProviderError(
                "Gemini provider is selected but GEMINI_API_KEY is not configured.",
                status_code=400
            )
        return GeminiProvider()

    elif provider_name == "openai":
        if not settings.OPENAI_API_KEY or not settings.OPENAI_API_KEY.strip():
            raise ProviderError(
                "OpenAI provider is selected but OPENAI_API_KEY is not configured.",
                status_code=400
            )
        return OpenAIProvider()

    else:
        raise ProviderError(
            f"Unsupported AI provider configuration: '{settings.AI_PROVIDER}'. "
            f"Allowed values are 'mock', 'ollama', 'gemini', 'openai'.",
            status_code=400
        )
