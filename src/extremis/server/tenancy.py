"""Tenant isolation for the hosted server.

Two layers, mirroring the pattern proven out in aradus-memory:

1. **Per-tenant storage path.** Every tenant gets its own data directory
   under EXTREMIS_SERVER_HOME/tenants/<slug>/. sqlite files, log dirs,
   chroma paths, etc. live there. Defense-in-depth: even if a query
   forgot its `WHERE namespace = ?` clause, cross-tenant reads can't
   happen because the file isn't shared.

2. **Canonical namespace builder + validation.** `tenant_namespace(uuid)`
   is the only legitimate way to construct a server-side namespace
   string. `assert_canonical_namespace()` raises TenantIsolationError
   on anything else — meant for the server boundary, not the OSS
   library (local users still pick whatever namespace string they
   like).
"""

from __future__ import annotations

import re
from pathlib import Path
from uuid import UUID

NAMESPACE_PREFIX = "extremis"
_NAMESPACE_RE = re.compile(
    rf"^{NAMESPACE_PREFIX}:[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}$"
)
_SLUG_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")


class TenantIsolationError(RuntimeError):
    """Raised when a server-side call presents a non-canonical namespace."""


def tenant_namespace(tenant_id: UUID | str) -> str:
    """The only legitimate way to construct a server-side namespace.

    Local OSS users can pick anything for Config.namespace; hosted-server
    callers must route through this helper so isolation is enforced at the
    boundary.
    """
    if isinstance(tenant_id, str):
        tenant_id = UUID(tenant_id)
    return f"{NAMESPACE_PREFIX}:{tenant_id}"


def is_canonical_namespace(namespace: str) -> bool:
    return bool(_NAMESPACE_RE.match(namespace or ""))


def assert_canonical_namespace(namespace: str) -> None:
    if not is_canonical_namespace(namespace):
        raise TenantIsolationError(
            f"refusing server call with non-canonical namespace {namespace!r}; "
            "use tenant_namespace(<uuid>) at the boundary."
        )


def slug_for_path(namespace: str) -> str:
    """Filesystem-safe slug derived from a namespace.

    Keeps canonical extremis:<uuid> namespaces readable in `ls` output
    while sanitising legacy/custom values so they can't escape the
    tenants/ directory.
    """
    slug = _SLUG_SAFE_RE.sub("_", namespace).strip("_")
    return slug or "unnamed"


def tenant_home(server_home: str | Path, namespace: str) -> Path:
    """Resolve the per-tenant data directory under the server home."""
    return Path(server_home).expanduser() / "tenants" / slug_for_path(namespace)
