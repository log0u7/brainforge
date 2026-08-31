from pathlib import Path

from brainforge.config.models import Config, ModelDef
from brainforge.providers.base import Provider
from brainforge.providers.cache import CacheProvider
from brainforge.providers.registry import build_provider


class ResolvedModel:
    def __init__(self, name: str, definition: ModelDef, provider: Provider):
        self.name = name
        self.definition = definition
        self.provider = provider


class ModelRegistry:
    def __init__(self, config: Config, cache_path: Path | str | None = None):
        self.config = config
        self.cache_path = Path(cache_path) if cache_path else None
        self._providers: dict[str, Provider] = {}
        self._resolved: dict[str, ResolvedModel] = {}

    def provider_for(self, provider_name: str) -> Provider:
        if provider_name not in self._providers:
            provider = build_provider(provider_name, self.config.providers[provider_name])
            if self.cache_path is not None:
                provider = CacheProvider(provider, self.cache_path / f"{provider_name}.sqlite")
            self._providers[provider_name] = provider
        return self._providers[provider_name]

    def resolve(self, model_name: str) -> ResolvedModel:
        if model_name not in self._resolved:
            definition = self.config.models[model_name]
            provider = self.provider_for(definition.provider)
            self._resolved[model_name] = ResolvedModel(model_name, definition, provider)
        return self._resolved[model_name]

    def names(self) -> list[str]:
        return sorted(self.config.models)
