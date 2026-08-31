from brainforge.domains.base import DomainPack, GateResult
from brainforge.domains.coding.pack import CodingPack
from brainforge.domains.generic import GenericPack
from brainforge.domains.security.pack import SecurityPack

_PACKS: dict[str, DomainPack] = {}


def register(pack: DomainPack) -> None:
    _PACKS[pack.name] = pack


def get_pack(name: str) -> DomainPack | None:
    return _PACKS.get(name)


def pack_names() -> list[str]:
    return sorted(_PACKS)


def get_or_generic(name: str) -> DomainPack:
    pack = _PACKS.get(name)
    if pack is not None:
        return pack
    return GenericPack(name=name, description=f"Auto-generated pack for domain '{name}'.")


register(SecurityPack())
register(CodingPack())

__all__ = [
    "DomainPack",
    "GateResult",
    "GenericPack",
    "get_or_generic",
    "get_pack",
    "pack_names",
    "register",
]
