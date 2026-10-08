from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO

from ._file import (
    _FileIdentity,
    _load_file_position,
    _open_file_identity,
    _save_file_position,
    _validate_open_file,
)
from ._parquet import (
    _normalize_columns,
    _parquet_layout,
    _ParquetCursor,
    _ParquetFile,
    _read_row_groups,
)
from .cardinality import Exact
from .cursor import Cursor, State
from .dataset import RangeDataset


def _pyarrow_parquet() -> Any:
    try:
        return import_module("pyarrow.parquet")
    except ModuleNotFoundError as error:
        if error.name == "pyarrow":
            raise ModuleNotFoundError(
                "ParquetDataset requires the 'parquet' extra: "
                "uv add 'rillway[parquet]'"
            ) from error
        raise


@dataclass(init=False)
class ParquetDataset(RangeDataset[dict[str, object]]):
    path: str
    columns: tuple[str, ...] | None
    _files: tuple[_ParquetFile, ...]
    _file_ends: tuple[int, ...]
    _length: int
    _identity: _FileIdentity

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        columns: Iterable[str] | None = None,
    ):
        normalized_columns = _normalize_columns(columns)
        normalized_path = str(Path(path).absolute())
        parquet_module = _pyarrow_parquet()
        with open(normalized_path, "rb") as reader:
            identity = _open_file_identity(reader)
            row_groups = _read_row_groups(parquet_module, reader)
        files = (_ParquetFile(normalized_path, row_groups),)
        file_ends, length = _parquet_layout(files)

        self.path = normalized_path
        self.columns = normalized_columns
        self._files = files
        self._file_ends = file_ends
        self._length = length
        self._identity = identity

    @property
    def cardinality(self) -> Exact:
        return Exact(self._length)

    @property
    def description(self) -> str:
        return f"Parquet(path={self.path!r}, columns={self.columns!r})"

    def _open_range(
        self,
        start: int,
        stop: int,
    ) -> Cursor[dict[str, object]]:
        return _ParquetCursor(self, start, stop)

    def _parquet_module(self) -> Any:
        return _pyarrow_parquet()

    @contextmanager
    def _open_parquet_file(
        self,
        file: _ParquetFile,
    ) -> Generator[BinaryIO]:
        with open(file.path, "rb") as reader:
            _validate_open_file(reader, self._identity)
            yield reader

    def _snapshot_parquet(self, position: int) -> State:
        state, _ = _save_file_position(
            self.path,
            None,
            self._identity,
            position,
        )
        return state

    def _restore_parquet(self, state: State, start: int, stop: int) -> int:
        position, identity = _load_file_position(state, self.path)
        if not start <= position <= stop:
            raise ValueError("Parquet checkpoint position is outside the requested range")
        if identity != self._identity:
            raise ValueError("Parquet checkpoint belongs to a different source file")
        return position
