from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}


def _assert_supported_suffix(path: Path) -> None:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file extension: {path.suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    _assert_supported_suffix(table_path)
    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path)
    return pd.read_excel(table_path)


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_supported_suffix(out_path)
    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(out_path, index=index)
    else:
        df.to_excel(out_path, index=index)
    return out_path
