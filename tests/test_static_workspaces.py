from __future__ import annotations

import pytest
import torch

from gcw.performance.workspaces import (
    StaticTensorSpec,
    StaticWorkspacePool,
    WorkspaceSignature,
)


def _signature(name: str = "pair") -> WorkspaceSignature:
    return WorkspaceSignature(
        name,
        "cpu",
        (
            ("mask", StaticTensorSpec((2, 4), torch.bool)),
            ("tokens", StaticTensorSpec((2, 4), torch.long)),
        ),
    )


def test_static_workspace_stages_exact_signature_and_blocks_early_reuse() -> None:
    signature = _signature()
    pool = StaticWorkspacePool((signature,), lanes=2)
    lease = pool.acquire(signature, 0)
    staged = lease.stage(
        {
            "tokens": torch.arange(8).reshape(2, 4),
            "mask": torch.ones(2, 4, dtype=torch.bool),
        }
    )
    assert torch.equal(staged["tokens"], torch.arange(8).reshape(2, 4))
    with pytest.raises(RuntimeError, match="active forward/backward"):
        pool.acquire(signature, 0)
    independent = pool.acquire(signature, 1)
    independent.release_after_backward()
    lease.release_after_backward()
    with pytest.raises(RuntimeError, match="released"):
        _ = lease.tensors
    pool.acquire(signature, 0).release_after_backward()


def test_static_workspace_fails_closed_for_shapes_names_and_signatures() -> None:
    signature = _signature()
    pool = StaticWorkspacePool((signature,), lanes=1)
    with pool.acquire(signature, 0) as lease:
        with pytest.raises(ValueError, match="names differ"):
            lease.stage({"tokens": torch.zeros(2, 4, dtype=torch.long)})
        with pytest.raises(ValueError, match="differs"):
            lease.stage(
                {
                    "tokens": torch.zeros(2, 5, dtype=torch.long),
                    "mask": torch.ones(2, 4, dtype=torch.bool),
                }
            )
    with pytest.raises(RuntimeError, match="unseen"):
        pool.acquire(_signature("other"), 0)
    with pytest.raises(ValueError, match="outside"):
        pool.acquire(signature, 1)
