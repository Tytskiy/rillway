import pytest

from rillway import Exact, ParquetDataset, RangeDataset

pyarrow = pytest.importorskip("pyarrow")
parquet = pytest.importorskip("pyarrow.parquet")


def write_records(path, records):
    parquet.write_table(pyarrow.Table.from_pylist(records), path, row_group_size=2)


def test_parquet_is_a_replayable_range_dataset(tmp_path):
    path = tmp_path / "records.parquet"
    records = [
        {"id": 1, "text": "one"},
        {"id": 2, "text": "two"},
        {"id": 3, "text": "three"},
        {"id": 4, "text": "four"},
        {"id": 5, "text": "five"},
    ]
    write_records(path, records)
    dataset = ParquetDataset(path)

    assert isinstance(dataset, RangeDataset)
    assert dataset.cardinality == Exact(5)
    assert list(dataset) == records
    assert list(dataset) == records
    assert list(dataset.open_range(1, 4)) == records[1:4]
    assert list(dataset.shard(1, 2)) == records[2:]
    assert dataset.explain().startswith("Parquet(path=")


def test_parquet_projects_columns(tmp_path):
    path = tmp_path / "records.parquet"
    write_records(path, [{"id": 1, "text": "one"}, {"id": 2, "text": "two"}])
    dataset = ParquetDataset(path, columns=("text",))

    assert list(dataset) == [{"text": "one"}, {"text": "two"}]
    assert "columns=('text',)" in dataset.explain()


def test_parquet_checkpoint_resumes_across_row_groups(tmp_path):
    path = tmp_path / "records.parquet"
    records = [{"id": index} for index in range(6)]
    write_records(path, records)
    dataset = ParquetDataset(path)
    cursor = dataset.open_range(1, 6)

    assert next(cursor) == records[1]
    assert next(cursor) == records[2]
    checkpoint = cursor.state_dict()

    resumed = dataset.open_range(1, 6)
    resumed.load_state_dict(checkpoint)
    assert list(resumed) == records[3:]


def test_parquet_checkpoint_rejects_a_changed_file(tmp_path):
    path = tmp_path / "records.parquet"
    write_records(path, [{"id": 1}, {"id": 2}])
    dataset = ParquetDataset(path)
    checkpoint = dataset.cursor().state_dict()
    write_records(path, [{"id": 3}, {"id": 4}, {"id": 5}])

    with pytest.raises(ValueError, match="source file changed"):
        dataset.cursor().load_state_dict(checkpoint)


@pytest.mark.parametrize("columns", [(), ("id", "id"), "id"])
def test_parquet_validates_columns(tmp_path, columns):
    path = tmp_path / "records.parquet"
    write_records(path, [{"id": 1}])

    with pytest.raises((TypeError, ValueError)):
        ParquetDataset(path, columns=columns)
