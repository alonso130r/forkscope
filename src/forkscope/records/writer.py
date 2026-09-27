import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from forkscope.results.records import EpisodeRecord


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_jsonl(records: list[EpisodeRecord], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            row = _json_value(asdict(record))
            f.write(json.dumps(row, allow_nan=False) + "\n")
