from dataclasses import dataclass, field
from operator import index as to_index
from os import PathLike, fstat, stat
from typing import IO, Any, BinaryIO, Protocol, Self, TextIO, cast
from urllib.parse import urlsplit, urlunsplit

from upath import UPath

type _FilePath = str | PathLike[str] | UPath
type _LocalFileIdentity = tuple[int, int, int, int]
type _FileIdentity = str | _LocalFileIdentity


class _FileSystem(Protocol):
    def open(self, path: str, mode: str, **kwargs: Any) -> Any: ...

    def ukey(self, path: str) -> Any: ...

    def glob(self, path: str) -> list[str]: ...

    def unstrip_protocol(self, path: str) -> str: ...


@dataclass(frozen=True)
class _File:
    uri: str
    path: str = field(repr=False)
    filesystem: _FileSystem = field(repr=False, compare=False)
    local: bool = False

    @classmethod
    def from_path(cls, path: _FilePath) -> Self:
        universal = path if isinstance(path, UPath) else UPath(path)
        local = universal.protocol in {"", "file", "local"}
        if local:
            universal = universal.absolute()
        return cls(_redact_uri(str(universal)), universal.path, universal.fs, local)

    @classmethod
    def from_filesystem(
        cls,
        filesystem: _FileSystem,
        path: str,
        *,
        uri: str | None = None,
    ) -> Self:
        protocol = getattr(filesystem, "protocol", None)
        protocols = (protocol,) if isinstance(protocol, str) else protocol or ()
        return cls(
            _redact_uri(filesystem.unstrip_protocol(path) if uri is None else uri),
            path,
            filesystem,
            any(value in {"", "file", "local"} for value in protocols),
        )

    def open_binary(self) -> BinaryIO:
        return cast(BinaryIO, self.filesystem.open(self.path, "rb"))

    def open_text(self, *, encoding: str, newline: str) -> TextIO:
        return cast(
            TextIO,
            self.filesystem.open(
                self.path,
                "r",
                encoding=encoding,
                newline=newline,
            ),
        )


def _file_identity(
    file: _File,
    reader: IO[Any] | None = None,
) -> _FileIdentity:
    if file.local:
        info = stat(file.path) if reader is None else fstat(reader.fileno())
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
    try:
        identity = file.filesystem.ukey(file.path)
    except (AttributeError, NotImplementedError) as error:
        raise TypeError("filesystem does not provide stable file identities") from error
    if not isinstance(identity, str) or not identity:
        raise TypeError("filesystem does not provide stable file identities")
    return identity


def _validate_file(
    file: _File,
    expected: _FileIdentity | None,
    reader: IO[Any] | None = None,
) -> _FileIdentity:
    identity = _file_identity(file, reader)
    if expected is not None and identity != expected:
        raise ValueError("source file changed")
    return identity


def _save_file_position(
    file: _File,
    reader: IO[Any] | None,
    expected: _FileIdentity | None,
    position: int,
) -> tuple[dict[str, Any], _FileIdentity]:
    identity = _file_identity(file, reader)
    if expected is not None and identity != expected:
        raise RuntimeError("source file changed")
    return {"position": position, "source": identity}, identity


def _load_file_position(
    state: dict[str, Any],
    file: _File,
) -> tuple[int, _FileIdentity]:
    try:
        position = to_index(state["position"])
        identity = state["source"]
    except (KeyError, TypeError) as error:
        raise ValueError("invalid file checkpoint") from error
    if position < 0 or not isinstance(identity, (str, tuple)):
        raise ValueError("invalid file checkpoint")
    if _file_identity(file) != identity:
        raise ValueError("source file changed")
    return position, identity


def _redact_uri(uri: str) -> str:
    if "://" not in uri:
        return uri
    parts = urlsplit(uri)
    netloc = parts.netloc.rsplit("@", 1)[-1]
    query = "<redacted>" if parts.query else ""
    fragment = "<redacted>" if parts.fragment else ""
    return urlunsplit((parts.scheme, netloc, parts.path, query, fragment))
