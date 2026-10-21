"""Incremental CSV files that let an interrupted run resume where it stopped."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _cell(value) -> str:
    """CSV text of a value, as pandas would write it (missing values empty)."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if hasattr(value, "item") and not isinstance(value, np.ndarray):   # numpy scalars
        value = value.item()
        if isinstance(value, float) and value != value:
            return ""
    return str(value)


def _norm(value):
    """Normalize key values: a re-read CSV returns 6.0 instead of 6 and NaN instead of None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):                       # numpy scalars
        return _norm(value.item())
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


class CSVLog:
    """Append rows to a CSV file and remember which keys are already saved.

    Rows are written as soon as they are appended, so the file always holds
    every finished task. Rows may bring new columns; the file is then
    rewritten with the union of the columns.

    Args:
        path: CSV file (created with its parent folders if missing).
        key_cols: Columns that identify a task; a row whose key is already in
            the file is considered done.
    """

    def __init__(self, path: str | Path, key_cols: list[str]):
        self.path = Path(path)
        self.key_cols = list(key_cols)
        self._done: set[str] = set()
        self._columns: list[str] | None = None
        if self._exists():
            df = pd.read_csv(self.path, usecols=lambda c: c in self.key_cols)
            if set(self.key_cols) <= set(df.columns):
                self._done = {self._key(r) for r in df[self.key_cols].to_dict("records")}
            self._columns = list(pd.read_csv(self.path, nrows=0).columns)

    def _exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def _key(self, row: dict) -> str:
        return json.dumps([_norm(row.get(k)) for k in self.key_cols], default=str)

    def done(self, row: dict) -> bool:
        """Whether a row with the same key values is already saved."""
        return self._key(row) in self._done

    def append(self, rows: list[dict] | dict) -> None:
        """Append one row or a list of rows."""
        rows = [rows] if isinstance(rows, dict) else list(rows)
        if not rows:
            return
        columns = list(dict.fromkeys(k for r in rows for k in r))
        if self._columns is None or not self._exists():
            self._columns = columns
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows, columns=columns).to_csv(self.path, index=False)
        elif set(columns) - set(self._columns):
            # New columns: rewrite the file with the union of the columns (rare).
            combined = pd.concat([pd.read_csv(self.path), pd.DataFrame(rows)], ignore_index=True)
            combined.to_csv(self.path, index=False)
            self._columns = list(combined.columns)
        else:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows([[_cell(r.get(c)) for c in self._columns] for r in rows])
        self._done.update(self._key(r) for r in rows)

    def frame(self) -> pd.DataFrame:
        """The whole file (empty if it does not exist)."""
        return pd.read_csv(self.path) if self._exists() else pd.DataFrame(columns=self.key_cols)


def read_csv(path: str | Path) -> pd.DataFrame:
    """A CSV file, or an empty frame if it does not exist."""
    path = Path(path)
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def write_json(path: str | Path, data) -> None:
    """Write JSON (numpy values converted)."""
    def convert(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    Path(path).write_text(json.dumps(data, indent=2, default=convert))


def read_json(path: str | Path, default=None):
    """Read JSON, or return ``default`` if the file does not exist."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default
