class BrainforgeError(Exception):
    pass


class ConfigError(BrainforgeError):
    pass


class ProviderError(BrainforgeError):
    pass


class ProviderNotSupportedError(ProviderError):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class QualityGateError(BrainforgeError):
    pass
