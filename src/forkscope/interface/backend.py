"""Array operations for the backend selected by a benchmark run."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import Any, Literal, Protocol

import numpy as np

BackendName = Literal["numpy", "jax", "torch"]


class ArrayBackend(Protocol):
    name: BackendName

    def is_array(self, value: Any) -> bool: ...

    def metadata(self, value: Any) -> tuple[tuple[int, ...], Any, Any]: ...

    def copy(self, value: Any) -> Any: ...

    def add_batch_dim(self, value: Any) -> Any: ...

    def stack(self, values: list[Any], *, axis: int = 0) -> Any: ...

    def select(self, value: Any, index: int) -> Any: ...

    def to_host(self, value: Any) -> Any: ...


class NumPyBackend:
    name: BackendName = "numpy"

    def is_array(self, value: Any) -> bool:
        return isinstance(value, np.ndarray)

    def metadata(self, value: Any) -> tuple[tuple[int, ...], Any, Any]:
        return value.shape, value.dtype, None

    def copy(self, value: Any) -> Any:
        return value.copy()

    def add_batch_dim(self, value: Any) -> Any:
        return np.expand_dims(value, axis=0)

    def stack(self, values: list[Any], *, axis: int = 0) -> Any:
        return np.stack(values, axis=axis)

    def select(self, value: Any, index: int) -> Any:
        return value[index].copy()

    def to_host(self, value: Any) -> Any:
        return value.tolist()


class JAXBackend:
    name: BackendName = "jax"

    def __init__(self) -> None:
        try:
            self._jax = import_module("jax")
            self._jnp = import_module("jax.numpy")
        except ImportError as exc:
            raise ImportError("JAX backend requires 'forkscope[jax]'.") from exc

    def is_array(self, value: Any) -> bool:
        return isinstance(value, self._jax.Array)

    def metadata(self, value: Any) -> tuple[tuple[int, ...], Any, Any]:
        return value.shape, value.dtype, value.device

    def copy(self, value: Any) -> Any:
        return self._jnp.copy(value)

    def add_batch_dim(self, value: Any) -> Any:
        return self._jnp.expand_dims(value, axis=0)

    def stack(self, values: list[Any], *, axis: int = 0) -> Any:
        return self._jnp.stack(values, axis=axis)

    def select(self, value: Any, index: int) -> Any:
        return self._jnp.copy(value[index])

    def to_host(self, value: Any) -> Any:
        return np.asarray(value).tolist()


class TorchBackend:
    name: BackendName = "torch"

    def __init__(self) -> None:
        try:
            self._torch = import_module("torch")
        except ImportError as exc:
            raise ImportError("PyTorch backend requires 'forkscope[torch]'.") from exc

    def is_array(self, value: Any) -> bool:
        return isinstance(value, self._torch.Tensor)

    def metadata(self, value: Any) -> tuple[tuple[int, ...], Any, Any]:
        return tuple(value.shape), value.dtype, value.device

    def copy(self, value: Any) -> Any:
        return value.clone()

    def add_batch_dim(self, value: Any) -> Any:
        return value.unsqueeze(0)

    def stack(self, values: list[Any], *, axis: int = 0) -> Any:
        return self._torch.stack(values, dim=axis)

    def select(self, value: Any, index: int) -> Any:
        return value[index].clone()

    def to_host(self, value: Any) -> Any:
        return value.detach().cpu().tolist()


@cache
def resolve_backend(name: BackendName) -> ArrayBackend:
    """Load a requested backend without importing other optional frameworks."""
    if name == "numpy":
        return NumPyBackend()
    if name == "jax":
        return JAXBackend()
    if name == "torch":
        return TorchBackend()
    raise ValueError(f"Unknown backend {name!r}; choose 'numpy', 'jax', or 'torch'.")


def is_array_like(value: Any) -> bool:
    """Recognize array leaves without importing optional frameworks."""
    return not isinstance(value, np.generic) and all(
        hasattr(value, attribute) for attribute in ("shape", "dtype", "ndim")
    )


def validate_array_backend(
    value: Any, adapter: ArrayBackend, *, name: str, path: str = "$"
) -> None:
    """Reject arrays from another backend within a nested boundary value."""
    if adapter.is_array(value):
        return
    if is_array_like(value):
        raise TypeError(
            f"{name} at {path} contains {type(value).__name__}; "
            f"expected {adapter.name} array."
        )
    if isinstance(value, dict):
        for key, item in value.items():
            validate_array_backend(item, adapter, name=name, path=f"{path}[{key!r}]")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            validate_array_backend(item, adapter, name=name, path=f"{path}[{index}]")
