"""Stable identities for deployed learned components."""

from __future__ import annotations

import hashlib
import json


def selector_deployment_digest(payload):
    """Hash the exact REG selector deployment weights and inference config."""

    state = payload.get(
        "deployment_state_dict",
        payload.get("ema_network_state_dict", payload.get("network_state_dict")),
    )
    if state is None:
        raise ValueError("selector checkpoint has no deployment state")
    digest = hashlib.sha256()
    metadata = {
        "selector_architecture": payload.get("selector_architecture"),
        "selector_feature_version": payload.get("selector_feature_version"),
        "config": payload.get("config"),
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    )
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


__all__ = ["selector_deployment_digest"]
