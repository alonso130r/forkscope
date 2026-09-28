import json
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forkscope import RolloutBenchmarker
from forkscope.evaluation.episode import run_episode
from forkscope.interface import backend as backend_module
from forkscope.interface.backend import BackendName
from forkscope.interface.tree import (
    TreeStructureError,
    tree_add_batch_dim,
    tree_copy,
    tree_stack,
    tree_unbatch,
    validate_tree,
)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_optional_backend_tree_and_benchmark(backend: BackendName, tmp_path: Path) -> None:
    array_module = pytest.importorskip("jax.numpy" if backend == "jax" else "torch")

    def make_array(values: list[float]) -> Any:
        if backend == "jax":
            return array_module.asarray(values)
        return array_module.tensor(values, requires_grad=True)

    source = {"value": make_array([1.0])}
    copied = tree_copy(source, backend=backend)
    batched = tree_add_batch_dim(source, backend=backend)
    validate_tree(batched, source, batched=True, backend=backend)
    stacked = tree_stack([source, copied], backend=backend)
    selected = tree_unbatch(stacked, 1, backend=backend)
    assert type(selected["value"]) is type(source["value"])
    assert selected["value"].shape == source["value"].shape

    class Environment:
        observation_space = None
        action_space = None

        def reset(self, *, seed: int | None = None, options: Any = None) -> Any:
            return {"value": make_array([1.0])}, {}

        def step(self, action: Any) -> Any:
            assert type(action) is type(source["value"])
            return {"value": make_array([2.0])}, 1.0, True, False, {"array": action}

    class Planner:
        def reset(self, *, seed: int) -> None:
            pass

        def plan(self, observation: Any, *, horizon: int) -> Any:
            return observation["value"] * horizon

    benchmarker = RolloutBenchmarker(
        Environment, Planner, seeds=[0], horizons=[2], max_steps=1, backend=backend
    )
    benchmarker.run()
    output = tmp_path / "records.jsonl"
    benchmarker.write_results(output)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["steps"][0]["observation"] == {"value": [1.0]}
    assert row["steps"][0]["action"] == [2.0]
    assert row["steps"][0]["info"]["array"] == [2.0]

    with pytest.raises(TreeStructureError, match="expected .* array"):
        tree_copy({"value": np.array([1.0])}, backend=backend)

    class WrongPlanner(Planner):
        def plan(self, observation: Any, *, horizon: int) -> Any:
            return np.array([1.0])

    with pytest.raises(TypeError, match="planner action at \\$"):
        run_episode(Environment(), WrongPlanner(), seed=0, horizon=1, max_steps=1, backend=backend)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_missing_optional_backend_has_install_hint(
    backend: BackendName, monkeypatch: pytest.MonkeyPatch
) -> None:

    def missing_optional(name: str) -> Any:
        if name == backend:
            raise ModuleNotFoundError(name)
        return import_module(name)

    with monkeypatch.context() as patcher:
        patcher.setattr(backend_module, "import_module", missing_optional)
        backend_module.resolve_backend.cache_clear()
        with pytest.raises(ImportError, match=rf"forkscope\[{backend}\]"):
            backend_module.resolve_backend(backend)
    backend_module.resolve_backend.cache_clear()
    assert backend_module.resolve_backend("numpy").name == "numpy"
