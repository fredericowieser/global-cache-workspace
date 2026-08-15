from __future__ import annotations

import pytest

from gcw.performance.activation_lifetime import (
    ActivationLifetimeRuntime,
    ActivationUse,
    plan_activation_lifetimes,
)


def _plan():
    return plan_activation_lifetimes(
        ("forward_0", "forward_1", "backward_0", "backward_1"),
        (
            ActivationUse("cache", "forward_0", ("forward_1", "backward_0")),
            ActivationUse("loss_0", "forward_0", ("backward_0",)),
            ActivationUse("loss_1", "forward_1", ("backward_1",)),
            ActivationUse("scratch", "forward_1", ()),
        ),
    )


def test_activation_plan_releases_only_after_last_declared_use() -> None:
    plan = _plan()
    assert plan.release_after("forward_0") == ()
    assert plan.release_after("forward_1") == ("scratch",)
    assert plan.release_after("backward_0") == ("cache", "loss_0")
    assert plan.release_after("backward_1") == ("loss_1",)

    runtime = ActivationLifetimeRuntime(plan)
    cache, loss_0, loss_1, scratch = object(), object(), object(), object()
    runtime.register("cache", cache)
    runtime.register("loss_0", loss_0)
    assert runtime.finish_phase("forward_0") == ()
    runtime.register("loss_1", loss_1)
    runtime.register("scratch", scratch)
    assert runtime.finish_phase("forward_1") == ("scratch",)
    assert runtime.get("cache") is cache
    assert runtime.finish_phase("backward_0") == ("cache", "loss_0")
    with pytest.raises(RuntimeError, match="already released"):
        runtime.get("cache")
    assert runtime.finish_phase("backward_1") == ("loss_1",)
    assert runtime.live_names == ()


def test_activation_runtime_fails_closed_on_order_missing_and_duplicate_values() -> None:
    runtime = ActivationLifetimeRuntime(_plan())
    with pytest.raises(RuntimeError, match="phase order"):
        runtime.finish_phase("forward_1")
    runtime.register("cache", object())
    with pytest.raises(RuntimeError, match="registered twice"):
        runtime.register("cache", object())
    with pytest.raises(RuntimeError, match="missing live"):
        runtime.finish_phase("forward_0")


def test_activation_plan_rejects_noncanonical_or_backward_uses() -> None:
    with pytest.raises(ValueError, match="sorted"):
        plan_activation_lifetimes(
            ("a", "b"),
            (ActivationUse("z", "a", ("b",)), ActivationUse("a", "a", ("b",))),
        )
    with pytest.raises(ValueError, match="before production"):
        plan_activation_lifetimes(("a", "b"), (ActivationUse("x", "b", ("a",)),))
