# Rillway

Rillway is a small Python library for building lazy, reusable data pipelines.
It keeps the familiar feel of Python iterators while making it easy to replay a
pipeline, split up a dataset, and resume interrupted work.

```python
from rillway import IndexedDataset

numbers = IndexedDataset.from_source(range(10))

dataset = (
    numbers
    .map(lambda value: value * 2)
    .filter(lambda value: value % 3 == 0)
    .batch(2)
)

assert list(dataset) == [(0, 6), (12, 18)]
assert list(dataset) == [(0, 6), (12, 18)]
```

Rillway requires Python 3.12 or newer.

## The basic idea

A dataset is a reusable description of some work. Creating one does not read
any data. Each time you iterate over it, Rillway creates a fresh cursor and
runs the pipeline from the beginning.

```python
dataset = IndexedDataset.from_source([1, 2, 3]).map(str)

assert list(dataset) == ["1", "2", "3"]
assert list(dataset) == ["1", "2", "3"]
```

A cursor is a single trip through that dataset. Use it directly when you want
to stop early or save your place:

```python
with dataset.cursor() as cursor:
    first = next(cursor)
```

The context manager makes sure anything opened by the pipeline is cleaned up.
Normal completion and errors are cleaned up as well.

## Working with datasets

Start with `IndexedDataset.from_source()` when your data supports `len()` and
indexing, such as a list or `range`.

```python
dataset = IndexedDataset.from_source([10, 20, 30, 40])

assert dataset[1] == 20
assert list(dataset[1:3]) == [20, 30]
assert list(dataset.shard(1, 2)) == [30, 40]
```

Rillway keeps useful abilities when an operation allows it. Mapping an indexed
dataset still gives you an indexed dataset. Filtering may change how many
items remain, so its result behaves like a stream instead.

You can inspect a pipeline without running it:

```python
print(dataset.map(str).filter(str.isdigit).explain())
```

## Resuming work

Some datasets can save their current position and continue with a new cursor.
Treat the saved state as opaque and restore it before reading from the new
cursor.

```python
dataset = IndexedDataset.from_source(range(6)).map(lambda value: value * 10)

with dataset.cursor() as cursor:
    assert next(cursor) == 0
    checkpoint = cursor.state_dict()

resumed = dataset.cursor()
resumed.load_state_dict(checkpoint)

assert list(resumed) == [10, 20, 30, 40, 50]
```

Use `dataset.checkpointable` to check whether a pipeline supports this.
Checkpoints belong to the pipeline that created them; Rillway rejects a
checkpoint when the pipeline does not match.

## Available operations

- `map`
- `filter`
- `flat_map`
- `take` and `skip`
- `batch`
- `concat`
- `zip`
- `shard` for datasets that can be split by range
- indexing and slicing for indexed datasets

## Development

```shell
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run mypy src/rillway
```
