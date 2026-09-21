from brainforge.config.models import ProviderConfig
from brainforge.errors import ConfigError
from brainforge.providers.base import Provider
from brainforge.providers.mock import MockProvider
from brainforge.providers.openai_compat import OpenAICompatProvider
from brainforge.types import ProviderType

_DEFAULT_BASE_URLS = {
    ProviderType.OPENROUTER: "https://openrouter.ai/api/v1",
    ProviderType.ZEN: "https://opencode.ai/zen/v1",
    ProviderType.MLGW: "http://localhost:8080/v1",
    ProviderType.LOCAL: "http://localhost:11434/v1",
}


def build_provider(name: str, config: ProviderConfig) -> Provider:
    if config.type == ProviderType.MOCK:
        return MockProvider(name, config)
    provider_cls = OpenAICompatProvider
    if config.type is ProviderType.OPENAI:
        return provider_cls(name, config)
    default = _DEFAULT_BASE_URLS.get(config.type)
    if default is None:
        raise ConfigError(f"unknown provider type '{config.type}'")
    return provider_cls(name, config.model_copy(update={"base_url": config.base_url or default}))
