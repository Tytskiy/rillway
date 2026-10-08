from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, cast

from ._file import (
    _FileIdentity,
    _load_file_position,
    _open_file_identity,
    _save_file_position,
    _validate_open_file,
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
    _length: int
    _row_groups: tuple[int, ...]
    _identity: _FileIdentity

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        columns: Iterable[str] | None = None,
    ):
        if isinstance(columns, str):
            raise TypeError("columns must be an iterable of strings")
        normalized_columns = None if columns is None else tuple(columns)
        if normalized_columns is not None:
            if not normalized_columns:
                raise ValueError("columns must not be empty")
            if any(not isinstance(column, str) for column in normalized_columns):
                raise TypeError("columns must contain only strings")
            if len(set(normalized_columns)) != len(normalized_columns):
                raise ValueError("columns must be unique")

        normalized_path = str(Path(path).absolute())
        parquet_module = _pyarrow_parquet()
        with open(normalized_path, "rb") as reader:
            identity = _open_file_identity(reader)
            parquet_file = parquet_module.ParquetFile(reader)
            try:
                metadata = parquet_file.metadata
                row_groups = tuple(
                    metadata.row_group(index).num_rows
                    for index in range(metadata.num_row_groups)
                )
                length = metadata.num_rows
            finally:
                parquet_file.close()

        self.path = normalized_path
        self.columns = normalized_columns
        self._length = length
        self._row_groups = row_groups
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


class _ParquetCursor(Cursor[dict[str, object]]):
    def __init__(self, dataset: ParquetDataset, start: int, stop: int):
        super().__init__()
        self._parquet_dataset = dataset
        self._start = start
        self._stop = stop
        self._position = start
        self._rows: Iterator[dict[str, object]] | None = None
        self._reader: BinaryIO | None = None

    def _open(self) -> None:
        reader = open(self._parquet_dataset.path, "rb")
        self.callback(reader.close)
        _validate_open_file(reader, self._parquet_dataset._identity)

        parquet_file = _pyarrow_parquet().ParquetFile(reader)
        self.callback(parquet_file.close)
        first, offset = self._locate(self._position)
        last, _ = self._locate(self._stop - 1)
        batches = parquet_file.iter_batches(
            row_groups=list(range(first, last + 1)),
            columns=(
                None
                if self._parquet_dataset.columns is None
                else list(self._parquet_dataset.columns)
            ),
        )
        self._rows = self._iter_rows(batches, offset)
        self._reader = reader

    def _locate(self, position: int) -> tuple[int, int]:
        row_group_start = 0
        for index, size in enumerate(self._parquet_dataset._row_groups):
            if position < row_group_start + size:
                return index, position - row_group_start
            row_group_start += size
        raise RuntimeError("Parquet metadata does not cover the requested range")

    @staticmethod
    def _iter_rows(batches: Iterable[Any], skip: int) -> Iterator[dict[str, object]]:
        for batch in batches:
            rows = cast(list[dict[str, object]], batch.to_pylist())
            for row in rows:
                if skip:
                    skip -= 1
                else:
                    yield row

    def _next(self) -> dict[str, object]:
        if self._position == self._stop:
            raise StopIteration
        if self._rows is None:
            self._open()
        assert self._rows is not None
        try:
            row = next(self._rows)
        except StopIteration as error:
            raise RuntimeError("Parquet range ended before its declared stop") from error
        self._position += 1
        return row

    def _snapshot(self) -> State:
        state, _ = _save_file_position(
            self._parquet_dataset.path,
            self._reader,
            self._parquet_dataset._identity,
            self._position,
        )
        return state

    def _restore(self, state: State) -> None:
        position, identity = _load_file_position(state, self._parquet_dataset.path)
        if not self._start <= position <= self._stop:
            raise ValueError("Parquet checkpoint position is outside the requested range")
        if identity != self._parquet_dataset._identity:
            raise ValueError("Parquet checkpoint belongs to a different source file")
        self._position = position
