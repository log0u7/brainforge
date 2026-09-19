import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from brainforge.types import ApiStyle, Mode, ProviderType, RoleKind

_KNOWLEDGE_CUTOFF_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ProviderType
    base_url: str | None = None
    api_key: str | None = None
    api_style: ApiStyle = ApiStyle.CHAT_COMPLETIONS
    timeout: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class PricingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class ModelDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    family: str | None = None
    knowledge_cutoff: str | None = None
    pricing: PricingConfig | None = None

    @field_validator("knowledge_cutoff")
    @classmethod
    def _validate_cutoff(cls, value: str | None) -> str | None:
        if value is not None and not _KNOWLEDGE_CUTOFF_RE.match(value):
            raise ValueError("knowledge_cutoff must be formatted as YYYY-MM")
        return value


class RoleDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    kind: RoleKind | None = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    system_prompt: str | None = None


class PipelineDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str
    steps: list[str] | None = None
    mode: Mode = Mode.STANDARD
    min_confidence: float = Field(default=0.5, ge=0, le=1)
    reject_unresolved: bool = True
    reject_recitation_risk: bool = False


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model: str = "Qwen/Qwen3-8B"
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, le=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation: int = Field(default=16, ge=1)
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    save_steps: int = Field(default=100, ge=1)
    seed: int = Field(default=42, ge=0)
    output_dir: str = "experiments"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderConfig] = Field(min_length=1)
    models: dict[str, ModelDef] = Field(min_length=1)
    roles: dict[str, RoleDef] = Field(min_length=1)
    pipelines: dict[str, PipelineDef] = Field(min_length=1)
    judge_independence: Literal["enforced", "warn", "off"] = "enforced"
    training: TrainingConfig = TrainingConfig()

    def infer_role_kind(self, role_name: str) -> RoleKind:
        role = self.roles[role_name]
        if role.kind is not None:
            return role.kind
        lowered = role_name.lower()
        if "judge" in lowered:
            return RoleKind.JUDGE
        if "critic" in lowered or "general" in lowered:
            return RoleKind.CRITIC
        return RoleKind.TEACHER

    def validate_references(self) -> list[str]:
        errors: list[str] = []
        for model_name, model in self.models.items():
            if model.provider not in self.providers:
                errors.append(
                    f"model '{model_name}' references unknown provider '{model.provider}'"
                )
        for role_name, role in self.roles.items():
            if role.model not in self.models:
                errors.append(f"role '{role_name}' references unknown model '{role.model}'")
        for pipeline_name, pipeline in self.pipelines.items():
            if not pipeline.domain:
                errors.append(f"pipeline '{pipeline_name}' has no domain")
            for step in pipeline.steps or []:
                if step not in self.roles:
                    errors.append(f"pipeline '{pipeline_name}' references unknown role '{step}'")
        return errors

    def check_judge_independence(self) -> list[str]:
        violations: list[str] = []
        judges = [name for name in self.roles if self.infer_role_kind(name) == RoleKind.JUDGE]
        teachers = [name for name in self.roles if name not in judges]
        for judge_name in judges:
            judge_model = self.models[self.roles[judge_name].model]
            for teacher_name in teachers:
                teacher_model = self.models[self.roles[teacher_name].model]
                if judge_model.provider == teacher_model.provider:
                    violations.append(
                        f"judge role '{judge_name}' shares provider "
                        f"'{judge_model.provider}' with role '{teacher_name}'"
                    )
                elif (
                    judge_model.family
                    and teacher_model.family
                    and judge_model.family.lower() == teacher_model.family.lower()
                ):
                    violations.append(
                        f"judge role '{judge_name}' shares model family "
                        f"'{judge_model.family}' with role '{teacher_name}'"
                    )
        return violations
