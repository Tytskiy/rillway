import pytest

from rillway import Dataset, Exact, IndexedDataset, Infinite


def resource_stream(values, closed):
    def factory():
        try:
            yield from values
        finally:
            closed.append(True)

    return Dataset.from_factory(factory)


def test_stream_shuffle_is_bounded_reproducible_and_preserves_cardinality():
    consumed = []

    def factory():
        for value in range(20):
            consumed.append(value)
            yield value

    source = Dataset.from_factory(factory, cardinality=Exact(20))
    shuffled = source.shuffle(4, seed=7)
    cursor = shuffled.cursor()

    first = next(cursor)
    assert first in range(4)
    assert consumed == [0, 1, 2, 3]

    result = [first, *cursor]
    assert sorted(result) == list(range(20))
    assert result == list(source.shuffle(4, seed=7))
    assert result != list(source.shuffle(4, seed=8))
    assert shuffled.cardinality == Exact(20)


def test_stream_shard_selects_round_robin_partitions():
    source = Dataset.from_factory(lambda: iter(range(10)), cardinality=Exact(10))
    shards = [source.shard(index, 3) for index in range(3)]

    assert [shard.cardinality for shard in shards] == [Exact(4), Exact(3), Exact(3)]
    assert [list(shard) for shard in shards] == [
        [0, 3, 6, 9],
        [1, 4, 7],
        [2, 5, 8],
    ]


def test_indexed_repeat_is_lazy_reproducible_and_shuffles_each_pass():
    source = IndexedDataset.from_source(range(17))
    repeated = source.repeat(3, shuffle=True)

    assert isinstance(repeated, IndexedDataset)
    assert repeated.cardinality == Exact(51)
    assert list(repeated) == list(source.repeat(3, shuffle=True, seed=42))
    assert list(repeated) == list(source.repeat(None, shuffle=True).take(51))
    assert list(repeated) != list(source.repeat(3, shuffle=True, seed=7))

    epochs = [tuple(repeated[start : start + 17]) for start in range(0, 51, 17)]
    assert all(sorted(epoch) == list(range(17)) for epoch in epochs)
    assert len(set(epochs)) == 3


def test_stream_repeat_reopens_the_parent_and_can_run_forever():
    opened = []

    def factory():
        opened.append(True)
        return iter([1, 2])

    stream = Dataset.from_factory(factory, cardinality=Exact(2))
    repeated = stream.repeat(3)

    assert repeated.cardinality == Exact(6)
    assert list(repeated) == [1, 2, 1, 2, 1, 2]
    assert len(opened) == 3

    endless = stream.repeat(None)
    assert endless.cardinality == Infinite()
    assert list(endless.take(5)) == [1, 2, 1, 2, 1]

    empty = Dataset.from_factory(lambda: iter(()), cardinality=Exact(0)).repeat(None)
    assert list(empty) == []


def test_repeat_validates_count_and_only_indexed_datasets_expose_shuffle():
    stream = Dataset.from_factory(lambda: iter([1]))

    with pytest.raises(ValueError, match="nonnegative"):
        stream.repeat(-1)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        stream.repeat(2, shuffle=True)


@pytest.mark.parametrize("index,count", [(-1, 2), (2, 2), (0, 0)])
def test_stream_shard_validates_coordinates(index, count):
    with pytest.raises(ValueError):
        Dataset.from_factory(lambda: iter([1])).shard(index, count)


def test_shuffle_checkpoint_replays_without_storing_buffered_values():
    dataset = (
        IndexedDataset.from_source(range(30))
        .filter(lambda value: True)
        .shuffle(5, seed=7)
    )
    cursor = dataset.cursor()
    assert len([next(cursor) for _ in range(9)]) == 9
    state = cursor.state_dict()
    expected = list(cursor)

    assert set(state["state"]) == {"parent", "replay"}
    assert state["state"]["replay"] == 9
    resumed = dataset.cursor()
    resumed.load_state_dict(state)
    assert list(resumed) == expected


def test_repeat_cursor_resumes_across_epochs():
    dataset = IndexedDataset.from_source(range(5)).repeat(None, shuffle=True)
    cursor = dataset.cursor()
    assert [next(cursor) for _ in range(7)]
    state = cursor.state_dict()
    expected = [next(cursor) for _ in range(8)]
    cursor.close()

    resumed = dataset.cursor()
    resumed.load_state_dict(state)
    assert [next(resumed) for _ in range(8)] == expected
    resumed.close()


def test_repeat_closes_each_parent_cursor():
    closed = []
    repeated = resource_stream([1], closed).repeat(2)

    assert list(repeated) == [1, 1]
    assert closed == [True, True]

    cursor = resource_stream([1, 2], closed).repeat(None).cursor()
    assert next(cursor) == 1
    cursor.close()
    assert closed == [True, True, True]
