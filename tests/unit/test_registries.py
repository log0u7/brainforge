import pytest

from brainforge.config.models import ProviderConfig
from brainforge.errors import ConfigError
from brainforge.models.registry import ModelRegistry
from brainforge.providers.mock import MockProvider
from brainforge.providers.registry import build_provider
from brainforge.roles.registry import RoleRegistry


def test_model_registry_resolve(config):
    registry = ModelRegistry(config)
    resolved = registry.resolve("security")
    assert resolved.name == "security"
    assert resolved.definition.model == "mock-security"
    assert isinstance(resolved.provider, MockProvider)
    assert registry.resolve("security") is resolved


def test_model_registry_names(config):
    registry = ModelRegistry(config)
    assert registry.names() == ["coding", "general", "judge", "security"]


def test_provider_memoization(config):
    registry = ModelRegistry(config)
    provider_a = registry.provider_for("teacher_a")
    provider_b = registry.provider_for("teacher_a")
    assert provider_a is provider_b


def test_build_provider_unknown_type():
    bogus = ProviderConfig.model_construct(type="ghost")
    with pytest.raises(ConfigError, match="unknown provider type"):
        build_provider("x", bogus)


def test_role_registry_resolve(config):
    models = ModelRegistry(config)
    roles = RoleRegistry(config, models)
    binding = roles.resolve("security_teacher")
    assert binding.kind.value == "teacher"
    assert binding.model_key == "security"
    assert binding.model_name == "mock-security"
    assert binding.temperature == 0.2
    assert isinstance(binding.provider, MockProvider)
    assert roles.resolve("security_teacher") is binding
