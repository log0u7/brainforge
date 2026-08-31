from brainforge.providers.openai_compat import OpenAICompatProvider


class LocalProvider(OpenAICompatProvider):
    default_base_url = "http://localhost:11434/v1"
