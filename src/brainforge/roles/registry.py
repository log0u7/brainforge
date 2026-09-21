from brainforge.config.models import Config, RoleDef
from brainforge.models.registry import ModelRegistry
from brainforge.providers.base import Provider
from brainforge.types import RoleKind


class RoleBinding:
    def __init__(
        self,
        role_name: str,
        definition: RoleDef,
        kind: RoleKind,
        model_key: str,
        model_name: str,
        provider: Provider,
    ):
        self.role_name = role_name
        self.definition = definition
        self.kind = kind
        self.model_key = model_key
        self.model_name = model_name
        self.provider = provider

    @property
    def temperature(self) -> float:
        return self.definition.temperature

    @property
    def top_p(self) -> float | None:
        return self.definition.top_p

    @property
    def max_tokens(self) -> int | None:
        return self.definition.max_tokens

    @property
    def system_prompt(self) -> str | None:
        return self.definition.system_prompt


class RoleRegistry:
    def __init__(self, config: Config, model_registry: ModelRegistry):
        self.config = config
        self.model_registry = model_registry
        self._bindings: dict[str, RoleBinding] = {}

    def resolve(self, role_name: str) -> RoleBinding:
        if role_name not in self._bindings:
            definition = self.config.roles[role_name]
            kind = self.config.infer_role_kind(role_name)
            model_key = definition.model
            resolved_model = self.model_registry.resolve(model_key)
            self._bindings[role_name] = RoleBinding(
                role_name,
                definition,
                kind,
                model_key,
                resolved_model.definition.model,
                resolved_model.provider,
            )
        return self._bindings[role_name]

    def role_names(self) -> list[str]:
        return sorted(self.config.roles)
