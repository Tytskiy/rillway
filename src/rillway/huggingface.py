from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, ClassVar

from ._file import _File, _file_identity
from ._parquet import (
    _normalize_columns,
    _parquet_layout,
    _parquet_path_fingerprint,
    _ParquetCursor,
    _ParquetFile,
    _read_row_groups,
)
from .cardinality import Exact
from .cursor import Cursor
from .dataset import RangeDataset


def _huggingface_dependencies() -> tuple[Any, Any]:
    try:
        hub = import_module("huggingface_hub")
        parquet = import_module("pyarrow.parquet")
    except ModuleNotFoundError as error:
        if error.name in {"huggingface_hub", "pyarrow"}:
            raise ModuleNotFoundError(
                "HuggingFaceDataset requires the 'huggingface' extra: "
                "uv add 'rillway[huggingface]'"
            ) from error
        raise
    return hub.HfFileSystem, parquet


def _huggingface_filesystem() -> Any:
    filesystem, _ = _huggingface_dependencies()
    return filesystem()


def _pyarrow_parquet() -> Any:
    _, parquet = _huggingface_dependencies()
    return parquet


@dataclass(init=False)
class HuggingFaceDataset(RangeDataset[dict[str, object]]):
    _accept_legacy_position_checkpoint: ClassVar[bool] = True

    repo_id: str
    config: str
    split: str
    revision: str
    columns: tuple[str, ...] | None
    _files: tuple[_ParquetFile, ...]
    _file_ends: tuple[int, ...]
    _length: int
    _fingerprint: str

    def __init__(
        self,
        repo_id: str,
        *,
        config: str = "default",
        split: str = "train",
        revision: str = "refs/convert/parquet",
        columns: Iterable[str] | None = None,
    ):
        for name, value in (
            ("repo_id", repo_id),
            ("config", config),
            ("split", split),
            ("revision", revision),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value:
                raise ValueError(f"{name} must not be empty")

        normalized_columns = _normalize_columns(columns)
        filesystem = _huggingface_filesystem()
        root = f"datasets/{repo_id}@{revision}"
        paths = sorted(filesystem.glob(f"{root}/{config}/{split}/*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"no Parquet export for {repo_id!r}, config {config!r}, split {split!r}"
            )

        parquet = _pyarrow_parquet()
        files = []
        for path in paths:
            info = filesystem.info(path, expand_info=True)
            commit = getattr(info.get("last_commit"), "oid", None)
            if not isinstance(commit, str) or not commit:
                raise RuntimeError(f"Hugging Face did not provide a revision for {path!r}")
            resolved = filesystem.resolve_path(path)
            pinned_path = f"datasets/{repo_id}@{commit}/{resolved.path_in_repo}"
            source = _File.from_filesystem(filesystem, pinned_path)
            with source.open_binary() as reader:
                identity = _file_identity(source, reader)
                row_groups = _read_row_groups(parquet, reader)
            files.append(_ParquetFile(source, identity, row_groups))

        file_ends, length = _parquet_layout(files)
        fingerprint = _parquet_path_fingerprint(files)

        self.repo_id = repo_id
        self.config = config
        self.split = split
        self.revision = revision
        self.columns = normalized_columns
        self._files = tuple(files)
        self._file_ends = file_ends
        self._length = length
        self._fingerprint = fingerprint

    @property
    def cardinality(self) -> Exact:
        return Exact(self._length)

    @property
    def description(self) -> str:
        return (
            f"HuggingFace(repo_id={self.repo_id!r}, config={self.config!r}, "
            f"split={self.split!r}, columns={self.columns!r}, "
            f"fingerprint={self._fingerprint!r})"
        )

    def _open_range(
        self,
        start: int,
        stop: int,
    ) -> Cursor[dict[str, object]]:
        return _ParquetCursor(self, start, stop)

    def _parquet_module(self) -> Any:
        return _pyarrow_parquet()
