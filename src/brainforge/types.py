from enum import StrEnum


class ProviderType(StrEnum):
    OPENROUTER = "openrouter"
    ZEN = "zen"
    MLGW = "mlgw"
    OPENAI = "openai"
    LOCAL = "local"
    MOCK = "mock"


class ApiStyle(StrEnum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    ANTHROPIC = "anthropic"


class Mode(StrEnum):
    CHEAP = "cheap"
    STANDARD = "standard"
    MAXIMUM = "maximum"


class RoleKind(StrEnum):
    TEACHER = "teacher"
    CRITIC = "critic"
    JUDGE = "judge"


class SplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    TEST_POSTCUTOFF = "test_postcutoff"
