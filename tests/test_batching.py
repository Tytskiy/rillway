import pytest

from rillway import Bounds, Dataset, Exact, IndexedDataset, Infinite, RangeDataset, Unknown


def test_batch_by_size_groups_consecutive_items_and_emits_final_batch():
    values = [("a", 3), ("b", 5), ("c", 4), ("d", 2), ("e", 7)]
    batches = IndexedDataset.from_source(values).batch_by_size(
        8,
        size=lambda item: item[1],
    )

    assert list(batches) == [
        (("a", 3), ("b", 5)),
        (("c", 4), ("d", 2)),
        (("e", 7),),
    ]


@pytest.mark.parametrize(
    "size,error,message",
    [
        (lambda _: 0, ValueError, "positive integer"),
        (lambda _: -1, ValueError, "positive integer"),
        (lambda _: 1.5, TypeError, "integer"),
        (lambda _: 4, ValueError, "exceeds max_size"),
    ],
)
def test_batch_by_size_validates_item_sizes(size, error, message):
    dataset = IndexedDataset.from_source(["item"]).batch_by_size(3, size=size)

    with pytest.raises(error, match=message):
        list(dataset)


def test_batch_by_size_requires_a_positive_max_size():
    with pytest.raises(ValueError, match="max_size must be positive"):
        IndexedDataset.from_source([1]).batch_by_size(0, size=lambda _: 1)


@pytest.mark.parametrize(
    "parent,expected",
    [
        (Exact(0), Exact(0)),
        (Exact(7), Bounds(3, 7)),
        (Bounds(2, 7), Bounds(1, 7)),
        (Infinite(), Infinite()),
        (Unknown(), Unknown()),
    ],
)
def test_batch_by_size_propagates_cardinality(parent, expected):
    dataset = Dataset.from_factory(lambda: (), cardinality=parent).batch_by_size(
        3,
        size=lambda _: 1,
    )

    assert dataset.cardinality == expected


def test_batch_by_size_always_returns_a_stream_dataset():
    dataset = IndexedDataset.from_source(range(3)).batch_by_size(2, size=lambda _: 1)

    assert isinstance(dataset, Dataset)
    assert not isinstance(dataset, RangeDataset)
    assert dataset.explain().startswith("BatchBySize(max_size=2, size=")


def test_batch_by_size_checkpoint_replays_held_lookahead():
    dataset = IndexedDataset.from_source([4, 3, 5]).batch_by_size(8, size=lambda value: value)
    cursor = dataset.cursor()

    assert next(cursor) == (4, 3)
    checkpoint = cursor.state_dict()
    expected = list(cursor)

    assert set(checkpoint["state"]) == {"parent", "replay"}
    assert checkpoint["state"]["replay"] == 2
    resumed = dataset.cursor()
    resumed.load_state_dict(checkpoint)
    assert list(resumed) == expected == [(5,)]
