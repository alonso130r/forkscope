from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from forkscope.interface.backend import ArrayBackend, BackendName, resolve_backend

Tree = Any


class TreeStructureError(ValueError):
    """Raised when trees differ in structure or contain unsupported leaves."""


def tree_copy(tree: Tree, *, backend: BackendName = "numpy") -> Tree:
    """Return a deep copy of a supported tree, copying every array leaf."""
    adapter = resolve_backend(backend)
    return _map_leaves(tree, lambda leaf, _path: adapter.copy(leaf), adapter)


def tree_add_batch_dim(tree: Tree, *, backend: BackendName = "numpy") -> Tree:
    """Add a leading batch dimension of size one to every array leaf."""
    adapter = resolve_backend(backend)
    return _map_leaves(tree, lambda leaf, _path: adapter.add_batch_dim(leaf), adapter)


def tree_stack(
    trees: Sequence[Tree], *, axis: int = 0, backend: BackendName = "numpy"
) -> Tree:
    """Stack equally structured trees along ``axis``.

    Each corresponding array leaf must have the same shape and dtype. The
    resulting leading dimension is the number of input trees when ``axis=0``.
    """
    if not trees:
        raise ValueError("Cannot stack an empty sequence of trees.")

    adapter = resolve_backend(backend)
    reference = trees[0]
    for index, tree in enumerate(trees[1:], start=1):
        assert_same_structure(
            reference, tree, left_name="trees[0]", right_name=f"trees[{index}]", backend=backend
        )

    leaf_maps = [_leaf_map(tree, adapter) for tree in trees]

    def stack_leaf(_reference_leaf: Any, path: str) -> Any:
        leaves = [leaves_by_path[path] for leaves_by_path in leaf_maps]
        first = leaves[0]
        first_shape, first_dtype, first_device = adapter.metadata(first)
        for index, leaf in enumerate(leaves[1:], start=1):
            shape, dtype, device = adapter.metadata(leaf)
            if shape != first_shape:
                raise TreeStructureError(
                    f"Array shape mismatch at {path}: trees[0] has {first_shape}, "
                    f"trees[{index}] has {shape}."
                )
            if dtype != first_dtype:
                raise TreeStructureError(
                    f"Array dtype mismatch at {path}: trees[0] has {first_dtype}, "
                    f"trees[{index}] has {dtype}."
                )
            if device != first_device:
                raise TreeStructureError(
                    f"Array device mismatch at {path}: trees[0] has {first_device}, "
                    f"trees[{index}] has {device}."
                )
        try:
            return adapter.stack(leaves, axis=axis)
        except (IndexError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Cannot stack arrays at {path} along axis {axis}.") from exc

    return _map_leaves(reference, stack_leaf, adapter)


def tree_unbatch(tree: Tree, index: int, *, backend: BackendName = "numpy") -> Tree:
    """Select one item from the leading batch dimension of every leaf."""
    adapter = resolve_backend(backend)
    batch_size: int | None = None

    def select(leaf: Any, path: str) -> Any:
        nonlocal batch_size
        if leaf.ndim == 0:
            raise TreeStructureError(f"Expected a batched array at {path}, got a scalar array.")
        if batch_size is None:
            batch_size = leaf.shape[0]
        elif leaf.shape[0] != batch_size:
            raise TreeStructureError(
                f"Inconsistent batch size at {path}: got {leaf.shape[0]}, expected {batch_size}."
            )
        if not -leaf.shape[0] <= index < leaf.shape[0]:
            raise IndexError(
                f"Batch index {index} is out of range for {path} with batch size {leaf.shape[0]}."
            )
        return adapter.select(leaf, index)

    return _map_leaves(tree, select, adapter)


def validate_tree(
    tree: Tree,
    template: Tree,
    *,
    batched: bool = False,
    check_dtype: bool = True,
    name: str = "tree",
    backend: BackendName = "numpy",
) -> None:
    """Validate structure and leaf metadata against ``template``.

    With ``batched=True``, each leaf in ``tree`` must have exactly one extra
    leading dimension relative to its corresponding template leaf. Batch sizes
    must agree across all leaves. The function raises ``TreeStructureError``
    with a path-specific explanation and otherwise returns ``None``.
    """
    adapter = resolve_backend(backend)
    assert_same_structure(template, tree, left_name="template", right_name=name, backend=backend)
    expected_batch_size: int | None = None

    def check(actual: Any, expected: Any, path: str) -> Any:
        nonlocal expected_batch_size
        if batched:
            if actual.ndim != expected.ndim + 1:
                raise TreeStructureError(
                    f"{name} array at {path} should have {expected.ndim + 1} dimensions "
                    f"(leading batch dimension plus template shape {expected.shape}), "
                    f"got shape {actual.shape}."
                )
            if actual.shape[1:] != expected.shape:
                raise TreeStructureError(
                    f"{name} array at {path} has shape {actual.shape[1:]}, "
                    f"expected {expected.shape} after the batch dimension."
                )
            batch_size = actual.shape[0]
            if expected_batch_size is None:
                expected_batch_size = batch_size
            elif batch_size != expected_batch_size:
                raise TreeStructureError(
                    f"{name} has inconsistent batch sizes: {path} has {batch_size}, "
                    f"expected {expected_batch_size}."
                )
        elif actual.shape != expected.shape:
            raise TreeStructureError(
                f"{name} array at {path} has shape {actual.shape}, expected {expected.shape}."
            )

        if check_dtype and actual.dtype != expected.dtype:
            raise TreeStructureError(
                f"{name} array at {path} has dtype {actual.dtype}, expected {expected.dtype}."
            )
        return actual

    _map_pair_leaves(tree, template, check, adapter)


def assert_same_structure(
    left: Tree,
    right: Tree,
    *,
    left_name: str = "left tree",
    right_name: str = "right tree",
    backend: BackendName = "numpy",
) -> None:
    """Raise ``TreeStructureError`` unless two trees have matching containers."""
    _assert_structure(
        left,
        right,
        path="$",
        left_name=left_name,
        right_name=right_name,
        adapter=resolve_backend(backend),
    )


def _map_leaves(tree: Tree, fn: Any, adapter: ArrayBackend, path: str = "$") -> Tree:
    if adapter.is_array(tree):
        return fn(tree, path)
    if isinstance(tree, dict):
        if not all(isinstance(key, str) for key in tree):
            raise TreeStructureError(f"Dictionary keys at {path} must all be strings.")
        return {
            key: _map_leaves(value, fn, adapter, _key_path(path, key))
            for key, value in tree.items()
        }
    if isinstance(tree, tuple):
        return tuple(
            _map_leaves(value, fn, adapter, f"{path}[{index}]")
            for index, value in enumerate(tree)
        )
    if isinstance(tree, list):
        return [
            _map_leaves(value, fn, adapter, f"{path}[{index}]")
            for index, value in enumerate(tree)
        ]
    raise TreeStructureError(
        f"Unsupported leaf at {path}: expected {adapter.name} array, dict, tuple, or list; "
        f"got {type(tree).__name__}."
    )


def _map_pair_leaves(
    left: Tree, right: Tree, fn: Any, adapter: ArrayBackend, path: str = "$"
) -> None:
    if adapter.is_array(left) and adapter.is_array(right):
        fn(left, right, path)
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if not all(isinstance(key, str) for key in left):
            raise TreeStructureError(f"Dictionary keys at {path} must all be strings.")
        for key in left:
            _map_pair_leaves(left[key], right[key], fn, adapter, _key_path(path, key))
        return
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _map_pair_leaves(left_item, right_item, fn, adapter, f"{path}[{index}]")
        return
    raise TreeStructureError(f"Unexpected tree structure mismatch at {path}.")


def _assert_structure(
    left: Tree,
    right: Tree,
    *,
    path: str,
    left_name: str,
    right_name: str,
    adapter: ArrayBackend,
) -> None:
    if adapter.is_array(left) and adapter.is_array(right):
        return
    for name, value in ((left_name, left), (right_name, right)):
        if _looks_like_array(value) and not adapter.is_array(value):
            raise TreeStructureError(
                f"{name} array at {path} uses {type(value).__name__}; "
                f"expected {adapter.name} array."
            )
    if isinstance(left, dict) and isinstance(right, dict):
        if not all(isinstance(key, str) for key in left) or not all(
            isinstance(key, str) for key in right
        ):
            raise TreeStructureError(f"Dictionary keys at {path} must all be strings.")
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            raise TreeStructureError(
                f"Dictionary keys differ at {path}: {left_name} has "
                f"{sorted(left_keys)}, {right_name} has {sorted(right_keys)}."
            )
        for key in left:
            _assert_structure(
                left[key],
                right[key],
                path=_key_path(path, key),
                left_name=left_name,
                right_name=right_name,
                adapter=adapter,
            )
        return
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        if len(left) != len(right):
            raise TreeStructureError(
                f"Sequence lengths differ at {path}: {left_name} has {len(left)}, "
                f"{right_name} has {len(right)}."
            )
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_structure(
                left_item,
                right_item,
                path=f"{path}[{index}]",
                left_name=left_name,
                right_name=right_name,
                adapter=adapter,
            )
        return
    raise TreeStructureError(
        f"Container structure differs at {path}: {left_name} has "
        f"{type(left).__name__}, {right_name} has {type(right).__name__}."
    )


def _leaf_map(tree: Tree, adapter: ArrayBackend, path: str = "$") -> dict[str, Any]:
    if adapter.is_array(tree):
        return {path: tree}
    leaves: dict[str, Any] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            leaves.update(_leaf_map(value, adapter, _key_path(path, key)))
    elif isinstance(tree, (tuple, list)):
        for index, value in enumerate(tree):
            leaves.update(_leaf_map(value, adapter, f"{path}[{index}]"))
    return leaves


def _looks_like_array(value: Any) -> bool:
    return all(hasattr(value, attribute) for attribute in ("shape", "dtype", "ndim"))


def _key_path(path: str, key: str) -> str:
    return f"{path}[{key!r}]"
