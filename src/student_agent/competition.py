from __future__ import annotations

from typing import Any

import httpx2

from .config import Settings


async def bootstrap_run(
    settings: Settings, *, variant_id: str, case_set_version: str
) -> dict[str, Any]:
    """Provision/refresh the team run that scopes MCP evidence access."""
    url = f"{settings.competition_api_url}/api/v2/runs"
    headers = {"Authorization": f"Bearer {settings.team_api_key}"}
    async with httpx2.AsyncClient(headers=headers, timeout=30) as client:
        response = await client.post(url, json={"variant_id": variant_id})
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"competition run bootstrap returned HTTP {response.status_code} without JSON"
        ) from exc
    if not 200 <= response.status_code < 300:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        raise RuntimeError(
            f"competition run bootstrap failed (HTTP {response.status_code}): "
            f"{detail or 'unknown error'}"
        )
    if not isinstance(payload, dict):
        raise RuntimeError("competition run bootstrap returned a non-object response")
    if payload.get("variant_id") != variant_id:
        raise RuntimeError("competition run bootstrap returned the wrong variant")
    remote_version = payload.get("case_set_version")
    if remote_version != case_set_version:
        raise RuntimeError(
            "local case-set version does not match the provisioned run: "
            f"local={case_set_version!r}, remote={remote_version!r}; download the run bundle"
        )
    return payload
