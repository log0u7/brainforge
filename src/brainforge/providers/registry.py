from brainforge.config.models import ProviderConfig
from brainforge.errors import ConfigError
from brainforge.providers.base import Provider
from brainforge.providers.local import LocalProvider
from brainforge.providers.mlgw import MLGWProvider
from brainforge.providers.mock import MockProvider
from brainforge.providers.openai_compat import OpenAICompatProvider
from brainforge.providers.openrouter import OpenRouterProvider
from brainforge.providers.zen import ZenProvider
from brainforge.types import ProviderType

_PROVIDER_CLASSES = {
    ProviderType.OPENROUTER: OpenRouterProvider,
    ProviderType.ZEN: ZenProvider,
    ProviderType.MLGW: MLGWProvider,
    ProviderType.OPENAI: OpenAICompatProvider,
    ProviderType.LOCAL: LocalProvider,
    ProviderType.MOCK: MockProvider,
}


def build_provider(name: str, config: ProviderConfig) -> Provider:
    provider_cls = _PROVIDER_CLASSES.get(config.type)
    if provider_cls is None:
        raise ConfigError(f"unknown provider type '{config.type}'")
    return provider_cls(name, config)
