from brainforge.providers.openai_compat import OpenAICompatProvider


class OpenRouterProvider(OpenAICompatProvider):
    default_base_url = "https://openrouter.ai/api/v1"
