import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forkscope.interface.backend import (
    ArrayBackend,
    BackendName,
    is_array_like,
    resolve_backend,
)
from forkscope.records.records import EpisodeRecord


def _json_value(value: Any, adapter: ArrayBackend, path: str = "$") -> Any:
    if adapter.is_array(value):
        return adapter.to_host(value)
    if isinstance(value, np.generic):
        return value.item()
    if is_array_like(value):
        raise TypeError(
            f"Record value at {path} contains {type(value).__name__}; "
            f"expected {adapter.name} array."
        )
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name), adapter, f"{path}.{field.name}")
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _json_value(item, adapter, f"{path}[{key!r}]")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item, adapter, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


def write_jsonl(
    records: list[EpisodeRecord], path: str | Path, *, backend: BackendName = "numpy"
) -> None:
    adapter = resolve_backend(backend)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            row = _json_value(record, adapter)
            f.write(json.dumps(row, allow_nan=False) + "\n")
