import pytest
from upath import UPath

from rillway import CsvDataset, Unknown


def test_csv_reads_a_configured_upath():
    root = UPath("memory://rillway-tests/csv", auto_mkdir=True)
    path = root / "records.csv"
    path.write_text("name\nalice\nbob\n", encoding="utf-8")

    dataset = CsvDataset(path)
    cursor = dataset.cursor()

    assert next(cursor) == {"name": "alice"}
    checkpoint = cursor.state_dict()
    resumed = dataset.cursor()
    resumed.load_state_dict(checkpoint)
    assert list(resumed) == [{"name": "bob"}]
    assert "memory://rillway-tests/csv/records.csv" in dataset.explain()
    assert "auto_mkdir" not in dataset.explain()


def test_csv_is_lazy_replayable_and_uses_its_header(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text('name,note\nalice,"hello\nworld"\nbob,plain\n', encoding="utf-8")
    dataset = CsvDataset(path)
    expected = [
        {"name": "alice", "note": "hello\nworld"},
        {"name": "bob", "note": "plain"},
    ]

    assert dataset.cardinality == Unknown()
    assert list(dataset) == expected
    assert list(dataset) == expected
    assert dataset.explain().startswith("Csv(path=")


def test_csv_checkpoint_resumes_at_a_logical_record(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text('name,note\nalice,"hello\nworld"\nbob,plain\n', encoding="utf-8")
    dataset = CsvDataset(path)
    cursor = dataset.cursor()

    assert next(cursor) == {"name": "alice", "note": "hello\nworld"}
    state = cursor.state_dict()
    expected = list(cursor)

    assert state["state"]["position"] > len("name,note\n")
    resumed = dataset.cursor()
    resumed.load_state_dict(state)
    assert list(resumed) == expected == [{"name": "bob", "note": "plain"}]


def test_csv_supports_an_explicit_delimiter(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text("name;age\nalice;30\n", encoding="utf-8")
    dataset = CsvDataset(path, delimiter=";")

    assert list(dataset) == [{"name": "alice", "age": "30"}]
    assert "delimiter=';'" in dataset.explain()

    checkpoint = dataset.cursor().state_dict()
    with pytest.raises(ValueError, match="does not match"):
        CsvDataset(path).cursor().load_state_dict(checkpoint)


def test_csv_supports_headerless_files_and_text_encodings(tmp_path):
    path = tmp_path / "records.csv"
    path.write_bytes("alice,Montréal\n".encode("latin-1"))
    dataset = CsvDataset(
        path,
        columns=("name", "city"),
        encoding="latin-1",
    )

    assert list(dataset) == [{"name": "alice", "city": "Montréal"}]
    assert "columns=('name', 'city')" in dataset.explain()
    assert "encoding='latin-1'" in dataset.explain()


def test_csv_checkpoint_rejects_a_changed_file(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text("name\nalice\n", encoding="utf-8")
    dataset = CsvDataset(path)
    checkpoint = dataset.cursor().state_dict()
    path.write_text("name\nbob\ncharlie\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source file changed"):
        dataset.cursor().load_state_dict(checkpoint)


@pytest.mark.parametrize("delimiter", ["", "::"])
def test_csv_requires_a_single_character_delimiter(tmp_path, delimiter):
    with pytest.raises(ValueError, match="one character"):
        CsvDataset(tmp_path / "records.csv", delimiter=delimiter)


@pytest.mark.parametrize("columns", [(), ("name", "name"), "name"])
def test_csv_validates_explicit_columns(tmp_path, columns):
    with pytest.raises((TypeError, ValueError)):
        CsvDataset(tmp_path / "records.csv", columns=columns)


@pytest.mark.parametrize(
    "contents,error",
    [
        ("name,name\na,b\n", "duplicate"),
        ("name,note\nalice\n", "does not match"),
    ],
)
def test_csv_rejects_ambiguous_records(tmp_path, contents, error):
    path = tmp_path / "invalid.csv"
    path.write_text(contents, encoding="utf-8")
    cursor = CsvDataset(path).cursor()

    with pytest.raises(ValueError, match=error):
        next(cursor)
    assert cursor.closed
