from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, ClassVar

from ..cardinality import Exact
from ..cursor import Cursor
from ..dataset import RangeDataset
from ._file import _File, _file_identity, _FilePath
from ._parquet import (
    _normalize_columns,
    _parquet_fingerprint,
    _parquet_layout,
    _ParquetCursor,
    _ParquetFile,
    _read_row_groups,
)


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
    _accept_legacy_position_checkpoint: ClassVar[bool] = False

    path: str
    columns: tuple[str, ...] | None
    _files: tuple[_ParquetFile, ...]
    _file_ends: tuple[int, ...]
    _length: int
    _fingerprint: str

    def __init__(
        self,
        path: _FilePath,
        *,
        columns: Iterable[str] | None = None,
    ):
        normalized_columns = _normalize_columns(columns)
        source = _File.from_path(path)
        parquet_module = _pyarrow_parquet()
        files = []
        for file in _expand_files(source):
            with file.open_binary() as reader:
                identity = _file_identity(file, reader)
                row_groups = _read_row_groups(parquet_module, reader)
            files.append(_ParquetFile(file, identity, row_groups))
        normalized_files = tuple(files)
        fingerprint = _parquet_fingerprint(normalized_files)
        file_ends, length = _parquet_layout(normalized_files)

        self.path = source.uri
        self.columns = normalized_columns
        self._files = normalized_files
        self._file_ends = file_ends
        self._length = length
        self._fingerprint = fingerprint

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


def _expand_files(source: _File) -> tuple[_File, ...]:
    if not any(character in source.path for character in "*?["):
        return (source,)
    paths = sorted(source.filesystem.glob(source.path))
    if not paths:
        raise FileNotFoundError(f"no files match {source.uri!r}")
    return tuple(
        _File.from_filesystem(source.filesystem, path)
        for path in paths
    )
