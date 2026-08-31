from brainforge.providers.openai_compat import OpenAICompatProvider


class MLGWProvider(OpenAICompatProvider):
    default_base_url = "http://localhost:8080/v1"
