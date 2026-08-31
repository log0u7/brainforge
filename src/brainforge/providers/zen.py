from brainforge.providers.openai_compat import OpenAICompatProvider


class ZenProvider(OpenAICompatProvider):
    default_base_url = "https://opencode.ai/zen/v1"
