import pytest

from rillway import Bounds, Dataset, Exact, IndexedDataset


def test_interleave_uses_round_robin_and_keeps_remaining_items():
    left = Dataset.from_factory(lambda: iter([1, 2, 3]), cardinality=Exact(3))
    right = Dataset.from_factory(lambda: iter([10, 20]), cardinality=Exact(2))
    interleaved = left.interleave(right)

    assert list(interleaved) == [1, 10, 2, 20, 3]
    assert interleaved.cardinality == Exact(5)
    assert interleaved.explain().startswith("Interleave(count=2)")
    assert left.interleave() is left


def test_mix_is_weighted_reproducible_and_keeps_remaining_items():
    left = IndexedDataset.from_source([("left", index) for index in range(40)])
    right = IndexedDataset.from_source([("right", index) for index in range(10)])
    mixed = left.mix(right, weights=(4, 1), seed=7)

    result = list(mixed)
    assert sorted(result) == sorted([*left, *right])
    assert result == list(left.mix(right, weights=(4, 1), seed=7))
    assert result != list(left.mix(right, weights=(4, 1), seed=8))
    assert sum(source == "left" for source, _ in result[:25]) > 15
    assert mixed.cardinality == Exact(50)
    assert mixed.explain().startswith("Mix(weights=(4.0, 1.0), seed=7)")


@pytest.mark.parametrize(
    "weights, error",
    [
        ((1,), "one value per dataset"),
        ((1, 0), "positive finite"),
        ((1, float("inf")), "positive finite"),
    ],
)
def test_mix_validates_weights(weights, error):
    left = IndexedDataset.from_source([1])
    right = IndexedDataset.from_source([2])

    with pytest.raises(ValueError, match=error):
        left.mix(right, weights=weights)


def test_concat_preserves_indexing_only_when_every_parent_is_indexed():
    left = IndexedDataset.from_source([1, 2])
    middle = IndexedDataset.from_source([])
    right = IndexedDataset.from_source([3, 4, 5])

    indexed = left.concat(middle, right)
    assert isinstance(indexed, IndexedDataset)
    assert indexed.cardinality == Exact(5)
    assert indexed[3] == 4
    assert list(indexed) == [1, 2, 3, 4, 5]

    streamed = left.concat(right.filter(lambda value: value != 4))
    assert isinstance(streamed, Dataset)
    assert streamed.cardinality == Bounds(2, 5)
    assert list(streamed) == [1, 2, 3, 5]


def test_stream_concat_opens_components_lazily():
    second_calls = []
    first = IndexedDataset.from_source([1]).filter(lambda value: True)
    second = (
        IndexedDataset.from_source([2])
        .filter(lambda value: True)
        .map(lambda value: second_calls.append(value) or value)
    )
    cursor = first.concat(second).cursor()

    assert next(cursor) == 1
    assert second_calls == []
    assert next(cursor) == 2
    assert second_calls == [2]
    cursor.close()


def test_nested_indexed_concat_reuses_cached_lengths():
    cardinality_calls = 0

    class CountingIndexed(IndexedDataset[int]):
        @property
        def cardinality(self):
            nonlocal cardinality_calls
            cardinality_calls += 1
            return Exact(1)

        @property
        def description(self):
            return "Counting"

        def _get(self, position):
            return position

    dataset = CountingIndexed()
    for _ in range(12):
        dataset = dataset.concat(dataset)

    cardinality_calls = 0
    assert dataset[0] == 0
    assert cardinality_calls == 0

    assert list(CountingIndexed().cursor()) == [0]
    assert cardinality_calls == 1


def test_zip_preserves_indexing_and_defaults_to_shortest():
    left = IndexedDataset.from_source([1, 2, 3])
    right = IndexedDataset.from_source(["a", "b"])
    zipped = left.zip(right)

    assert isinstance(zipped, IndexedDataset)
    assert zipped.cardinality == Exact(2)
    assert zipped[1] == (2, "b")
    assert list(zipped) == [(1, "a"), (2, "b")]


def test_strict_zip_rejects_known_and_runtime_length_mismatches():
    left = IndexedDataset.from_source([1, 2, 3])
    right = IndexedDataset.from_source(["a", "b"])
    with pytest.raises(ValueError, match="different lengths"):
        left.zip(right, strict=True)

    uncertain_left = left.filter(lambda value: value < 3)
    uncertain_right = right.filter(lambda value: True)
    assert list(uncertain_left.zip(uncertain_right, strict=True)) == [
        (1, "a"),
        (2, "b"),
    ]

    shorter = right.filter(lambda value: value == "a")
    cursor = uncertain_left.zip(shorter, strict=True).cursor()
    with pytest.raises(ValueError):
        list(cursor)
    assert cursor.closed


def test_interleave_checkpoint_preserves_each_parent_position():
    left = IndexedDataset.from_source([1]).filter(bool)
    right = IndexedDataset.from_source([10, 20, 30]).filter(bool)
    dataset = left.interleave(right)
    cursor = dataset.cursor()
    assert [next(cursor), next(cursor), next(cursor)] == [1, 10, 20]
    state = cursor.state_dict()
    expected = list(cursor)

    resumed = dataset.cursor()
    resumed.load_state_dict(state)
    assert list(resumed) == expected == [30]


def test_mix_checkpoint_preserves_selection_and_parent_positions():
    left = IndexedDataset.from_source(range(10)).filter(lambda value: True)
    right = IndexedDataset.from_source(range(100, 105)).filter(lambda value: True)
    dataset = left.mix(right, weights=(3, 1), seed=7)
    cursor = dataset.cursor()
    assert len([next(cursor) for _ in range(8)]) == 8
    state = cursor.state_dict()
    expected = list(cursor)

    assert set(state["state"]) == {"draw", "active", "parents"}
    resumed = dataset.cursor()
    resumed.load_state_dict(state)
    assert list(resumed) == expected


def test_concat_and_zip_checkpoint_their_parent_cursors():
    concatenated = (
        IndexedDataset.from_source([1, 2]).filter(bool).concat(IndexedDataset.from_source([3, 4]))
    )
    concat_cursor = concatenated.cursor()
    assert next(concat_cursor) == 1
    concat_state = concat_cursor.state_dict()
    resumed_concat = concatenated.cursor()
    resumed_concat.load_state_dict(concat_state)
    assert list(resumed_concat) == [2, 3, 4]

    zipped = (
        IndexedDataset.from_source([1, 2, 3])
        .filter(bool)
        .zip(IndexedDataset.from_source(["a", "b", "c"]))
    )
    zip_cursor = zipped.cursor()
    assert next(zip_cursor) == (1, "a")
    zip_state = zip_cursor.state_dict()
    resumed_zip = zipped.cursor()
    resumed_zip.load_state_dict(zip_state)
    assert list(resumed_zip) == [(2, "b"), (3, "c")]
