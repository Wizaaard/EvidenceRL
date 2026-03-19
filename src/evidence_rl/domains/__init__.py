"""Domain configurations for EvidenceRL.

Each domain bundles all task-specific settings so the core reward/retrieval/training
logic stays domain-agnostic.

Usage:
    from evidence_rl.domains import get_domain, MedicalDomain, ALCEDomain

    domain = get_domain("medical")   # or "alce"
    sections = domain.parse_context(patient_context)
"""

from .base import DomainConfig
from .medical import MedicalDomain
from .alce import ALCEDomain
from .barexam import BarExamDomain

_REGISTRY: dict[str, type[DomainConfig]] = {
    "medical": MedicalDomain,
    "alce": ALCEDomain,
    "barexam": BarExamDomain,
}


def get_domain(name: str, **kwargs) -> DomainConfig:
    """Instantiate a domain config by name.

    Args:
        name: Domain identifier ("medical", "alce")
        **kwargs: Passed to the domain constructor

    Returns:
        Configured DomainConfig instance
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown domain '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)


def list_domains() -> list[str]:
    """Return available domain names."""
    return list(_REGISTRY.keys())
