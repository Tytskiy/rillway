from pathlib import Path
from types import SimpleNamespace

import pytest

import rillway.huggingface as huggingface_module
from rillway import Exact, HuggingFaceDataset, RangeDataset

pyarrow = pytest.importorskip("pyarrow")
parquet = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("huggingface_hub")


class FakeHfFileSystem:
    def __init__(self, files: dict[str, Path]):
        self.files = files
        self.opened: list[str] = []

    def glob(self, pattern: str) -> list[str]:
        assert pattern.endswith("@refs/convert/parquet/default/train/*.parquet")
        root = pattern.removesuffix("*.parquet")
        return [f"{root}{name}" for name in sorted(self.files)]

    def info(self, path: str, *, expand_info: bool) -> dict[str, object]:
        assert expand_info
        name = path.rsplit("/", 1)[-1]
        commit = ("a" if name == "0000.parquet" else "b") * 40
        return {"last_commit": SimpleNamespace(oid=commit)}

    def resolve_path(self, path: str) -> SimpleNamespace:
        name = path.rsplit("/", 1)[-1]
        return SimpleNamespace(path_in_repo=f"default/train/{name}")

    def open(self, path: str, mode: str):
        assert mode == "rb"
        assert "@refs/convert/parquet" not in path
        self.opened.append(path)
        return self.files[path.rsplit("/", 1)[-1]].open(mode)


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    parquet.write_table(pyarrow.Table.from_pylist(records), path, row_group_size=2)


@pytest.fixture
def filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHfFileSystem:
    first = tmp_path / "0000.parquet"
    second = tmp_path / "0001.parquet"
    write_records(first, [{"id": 0, "text": "zero"}, {"id": 1, "text": "one"}])
    write_records(
        second,
        [
            {"id": 2, "text": "two"},
            {"id": 3, "text": "three"},
            {"id": 4, "text": "four"},
        ],
    )
    filesystem = FakeHfFileSystem({first.name: first, second.name: second})
    monkeypatch.setattr(
        huggingface_module,
        "_huggingface_filesystem",
        lambda: filesystem,
    )
    return filesystem


def test_huggingface_is_a_remote_parquet_range_dataset(
    filesystem: FakeHfFileSystem,
) -> None:
    dataset = HuggingFaceDataset(
        "example/data",
        columns=("text",),
    )

    assert isinstance(dataset, RangeDataset)
    assert dataset.cardinality == Exact(5)
    assert list(dataset) == [
        {"text": "zero"},
        {"text": "one"},
        {"text": "two"},
        {"text": "three"},
        {"text": "four"},
    ]
    assert list(dataset.open_range(1, 4)) == [
        {"text": "one"},
        {"text": "two"},
        {"text": "three"},
    ]
    assert list(dataset.shard(1, 2)) == [
        {"text": "two"},
        {"text": "three"},
        {"text": "four"},
    ]
    assert all("@refs/convert/parquet" not in path for path in filesystem.opened)
    assert dataset.explain().startswith("HuggingFace(repo_id='example/data'")


def test_huggingface_checkpoint_resumes_across_files(
    filesystem: FakeHfFileSystem,
) -> None:
    dataset = HuggingFaceDataset("example/data")
    cursor = dataset.open_range(1, 5)
    assert [next(cursor), next(cursor)] == [
        {"id": 1, "text": "one"},
        {"id": 2, "text": "two"},
    ]
    checkpoint = cursor.state_dict()

    resumed = dataset.open_range(1, 5)
    resumed.load_state_dict(checkpoint)
    assert list(resumed) == [
        {"id": 3, "text": "three"},
        {"id": 4, "text": "four"},
    ]


def test_huggingface_requires_a_parquet_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filesystem = FakeHfFileSystem({})
    monkeypatch.setattr(
        huggingface_module,
        "_huggingface_filesystem",
        lambda: filesystem,
    )

    with pytest.raises(FileNotFoundError, match="no Parquet export"):
        HuggingFaceDataset("example/data")
