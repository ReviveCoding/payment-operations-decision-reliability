"""Additive release-integrity hardening for the PaymentOps reliability framework."""

from payment_ops_hardening.contract_template import (
    ContractTemplateError,
    render_release_contract_template,
)
from payment_ops_hardening.release_security import (
    ReleaseSecurityError,
    finalize_release_security,
    verify_release_security,
)
from payment_ops_hardening.verified_loader import (
    read_verified_artifact,
    read_verified_artifacts,
)

__version__ = "0.9.4"

__all__ = [
    "ContractTemplateError",
    "render_release_contract_template",
    "ReleaseSecurityError",
    "finalize_release_security",
    "verify_release_security",
    "read_verified_artifact",
    "read_verified_artifacts",
]
