from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]
Tree = Array | dict[str, "Tree"] | tuple["Tree", ...] | list["Tree"]


class TreeStructureError(ValueError):
    """Raised when trees differ in structure or contain unsupported leaves."""


def tree_copy(tree: Tree) -> Tree:
    """Return a deep copy of a supported tree, copying every array leaf."""
    return _map_leaves(tree, lambda leaf, _path: leaf.copy())


def tree_add_batch_dim(tree: Tree) -> Tree:
    """Add a leading batch dimension of size one to every array leaf."""
    return _map_leaves(tree, lambda leaf, _path: np.expand_dims(leaf, axis=0))


def tree_stack(trees: Sequence[Tree], *, axis: int = 0) -> Tree:
    """Stack equally structured trees along ``axis``.

    Each corresponding array leaf must have the same shape and dtype. The
    resulting leading dimension is the number of input trees when ``axis=0``.
    """
    if not trees:
        raise ValueError("Cannot stack an empty sequence of trees.")

    reference = trees[0]
    for index, tree in enumerate(trees[1:], start=1):
        assert_same_structure(reference, tree, left_name="trees[0]", right_name=f"trees[{index}]")

    leaf_maps = [_leaf_map(tree) for tree in trees]

    def stack_leaf(_reference_leaf: Array, path: str) -> Array:
        leaves = [leaves_by_path[path] for leaves_by_path in leaf_maps]
        first = leaves[0]
        for index, leaf in enumerate(leaves[1:], start=1):
            if leaf.shape != first.shape:
                raise TreeStructureError(
                    f"Array shape mismatch at {path}: trees[0] has {first.shape}, "
                    f"trees[{index}] has {leaf.shape}."
                )
            if leaf.dtype != first.dtype:
                raise TreeStructureError(
                    f"Array dtype mismatch at {path}: trees[0] has {first.dtype}, "
                    f"trees[{index}] has {leaf.dtype}."
                )
        try:
            return np.stack(leaves, axis=axis)
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Cannot stack arrays at {path} along axis {axis}.") from exc

    return _map_leaves(reference, stack_leaf)


def tree_unbatch(tree: Tree, index: int) -> Tree:
    """Select one item from the leading batch dimension of every leaf."""
    batch_size: int | None = None

    def select(leaf: Array, path: str) -> Array:
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
        return leaf[index].copy()

    return _map_leaves(tree, select)


def validate_tree(
    tree: Tree,
    template: Tree,
    *,
    batched: bool = False,
    check_dtype: bool = True,
    name: str = "tree",
) -> None:
    """Validate structure and leaf metadata against ``template``.

    With ``batched=True``, each leaf in ``tree`` must have exactly one extra
    leading dimension relative to its corresponding template leaf. Batch sizes
    must agree across all leaves. The function raises ``TreeStructureError``
    with a path-specific explanation and otherwise returns ``None``.
    """
    assert_same_structure(template, tree, left_name="template", right_name=name)
    expected_batch_size: int | None = None

    def check(actual: Array, expected: Array, path: str) -> Array:
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

    _map_pair_leaves(tree, template, check)


def assert_same_structure(
    left: Tree,
    right: Tree,
    *,
    left_name: str = "left tree",
    right_name: str = "right tree",
) -> None:
    """Raise ``TreeStructureError`` unless two trees have matching containers."""
    _assert_structure(left, right, path="$", left_name=left_name, right_name=right_name)


def _map_leaves(tree: Tree, fn: Any, path: str = "$") -> Tree:
    if isinstance(tree, np.ndarray):
        return fn(tree, path)
    if isinstance(tree, dict):
        if not all(isinstance(key, str) for key in tree):
            raise TreeStructureError(f"Dictionary keys at {path} must all be strings.")
        return {key: _map_leaves(value, fn, _key_path(path, key)) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_map_leaves(value, fn, f"{path}[{index}]") for index, value in enumerate(tree))
    if isinstance(tree, list):
        return [_map_leaves(value, fn, f"{path}[{index}]") for index, value in enumerate(tree)]
    raise TreeStructureError(
        f"Unsupported leaf at {path}: expected numpy.ndarray, dict, tuple, or list; "
        f"got {type(tree).__name__}."
    )


def _map_pair_leaves(left: Tree, right: Tree, fn: Any, path: str = "$") -> None:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        fn(left, right, path)
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if not all(isinstance(key, str) for key in left):
            raise TreeStructureError(f"Dictionary keys at {path} must all be strings.")
        for key in left:
            _map_pair_leaves(left[key], right[key], fn, _key_path(path, key))
        return
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _map_pair_leaves(left_item, right_item, fn, f"{path}[{index}]")
        return
    raise TreeStructureError(f"Unexpected tree structure mismatch at {path}.")


def _assert_structure(
    left: Tree,
    right: Tree,
    *,
    path: str,
    left_name: str,
    right_name: str,
) -> None:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return
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
            )
        return
    raise TreeStructureError(
        f"Container structure differs at {path}: {left_name} has "
        f"{type(left).__name__}, {right_name} has {type(right).__name__}."
    )


def _leaf_map(tree: Tree, path: str = "$") -> dict[str, Array]:
    if isinstance(tree, np.ndarray):
        return {path: tree}
    leaves: dict[str, Array] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            leaves.update(_leaf_map(value, _key_path(path, key)))
    elif isinstance(tree, (tuple, list)):
        for index, value in enumerate(tree):
            leaves.update(_leaf_map(value, f"{path}[{index}]"))
    return leaves


def _key_path(path: str, key: str) -> str:
    return f"{path}[{key!r}]"
