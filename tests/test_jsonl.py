import json

import pytest

from rillway import JsonlDataset, Unknown


def test_jsonl_is_lazy_replayable_and_structured(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"id": 1}\n[2, 3]\nnull\n', encoding="utf-8")
    dataset = JsonlDataset(path)

    assert dataset.cardinality == Unknown()
    assert list(dataset) == [{"id": 1}, [2, 3], None]
    assert list(dataset) == [{"id": 1}, [2, 3], None]
    assert dataset.explain().startswith("Jsonl(path=")


def test_jsonl_checkpoint_uses_byte_offset(tmp_path):
    path = tmp_path / "records.jsonl"
    first = json.dumps({"text": "α"}, ensure_ascii=False) + "\n"
    path.write_text(first + '{"text": "β"}\n', encoding="utf-8")
    dataset = JsonlDataset(path)
    cursor = dataset.cursor()

    assert next(cursor) == {"text": "α"}
    state = cursor.state_dict()
    expected = list(cursor)

    assert state["state"]["offset"] == len(first.encode())
    assert "source" in state["state"]
    resumed = dataset.cursor()
    resumed.load_state_dict(state)
    assert list(resumed) == expected == [{"text": "β"}]


def test_jsonl_closes_on_missing_or_invalid_input(tmp_path):
    missing = JsonlDataset(tmp_path / "missing.jsonl").cursor()
    with pytest.raises(FileNotFoundError):
        next(missing)
    assert missing.closed

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("not-json\n", encoding="utf-8")
    cursor = JsonlDataset(invalid).cursor()
    with pytest.raises(json.JSONDecodeError):
        next(cursor)
    assert cursor.closed


def test_jsonl_checkpoint_rejects_a_changed_file(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"id": 1}\n', encoding="utf-8")
    dataset = JsonlDataset(path)
    checkpoint = dataset.cursor().state_dict()
    path.write_text('{"id": 2}\n{"id": 3}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="source file changed"):
        dataset.cursor().load_state_dict(checkpoint)
