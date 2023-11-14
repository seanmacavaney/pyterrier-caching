# Caching for PyTerrier

`pyterrier-caching` provides several components for caching intermediate results.

The right component will depend on your use case.

## Installation

Install this package using pip:

```bash
pip install pyterrier-caching
```

## Caching Results when Indexing

`IndexerCache` saves the sequence of documents encountered in an indexing pipeline.
It allows you to repeat that sequence, without needing to re-execute the computations
up to that point.

**Example use case:** I want to test how different retrieval engines perform over [learned
sparse representations](https://arxiv.org/abs/2303.13416), but I don't want to
re-compute the representations each time.

You use an `IndexerCache` the same way you would use an indexer: as the last component
of a pipeline. Rather than building an index of the data, the `IndexerCache` will save
your results to a file on disk. This file can be re-read by iterating over the cache
object with `iter(cache)`.

Example:

```python
import pyterrier as pt
pt.init()
from pyterrier_caching import IndexerCache

# Setup
cache = IndexerCache('path/to/cache')
dataset = pt.get_dataset('some-dataset') # e.g., 'irds:msmarco-passage'

# Use the IndexerCache cache object just as you would an indexer
cache_pipeline = MyExpensiveTransformer() >> cache

# The following line will save the results of MyExpensiveTransformer() to path/to/cache
cache_pipeline.index(dataset.get_corpus_iter())

# Now you can build multiple indexes over the results of MyExpensiveTransformer without
# needing to re-run it each time
indexer1 = ... # e.g., pt.IterDictIndexer('./path/to/index.terrier')
indexer1.index(iter(cache))
indexer2 = ... # e.g., pyterrier_pisa.PisaIndex('./path/to/index.pisa')
indexer2.index(iter(cache))
```

<details>
<summary>👁‍🗨 More Details</summary>

`IndexerCache` currently has one implementation, `Lz4PickleIndexerCache`, which is
set as the default. `Lz4PickleIndexerCache` saves the sequence as a sequence of
LZ4-compressed pickled dicts in the file: `data.pkl.lz4`. Byte-level offsets for each
document are stored as a numpy-compatible float64 array in `offsets.np`. If the `docno`
column is present, an [`npids`]() structure is also stored, facilitating reverse-lookups
of documents by their docno.

</details>


## Caching Results from a Scorer

`ScorerCache` saves the `score` based on `query` and `docno`. When the same
`query`-`docno` combination is encountered again, the value is read from the cache,
avoiding re-computation.

**Example use case:** I want to test a neural relevance model over several first-stage
retrieval models, but they bring back many of the same documents, so I don't want to
re-compute the scores each time.

You use a `ScorerCache` in place of the scorer in a pipeline. It holds a reference to
the scorer so that it can compute values that are missing from the cache. You need to
"build" a `ScorerCache` before you can use it, which creates an internal mapping between
the string `docno` and the integer indexes at which the scores values are stored.

**⚠️ Important Caveats**:
 - `ScorerCache` saves scores based on **only** the value of the `query` and `docno`. All
   other information is ignored (e.g., the text of the document).
 - `ScorerCache` only stores the result of the `score` column. All other outputs of the
   scorer will be discarded. (Rank is also outputed, but is caculated by `ScorerCache`,
   not the scorer.)
 - Due to limitations of HDF5, only a single process can have the cache open for writing
   at a time.
 - Scores are saved as `float32` values. Other values will attempted to be cast up/down
   to float32.
 - The value of `nan` is reserved as an indicator that the value is missing from the cache.
   Scorers should not return this value.
 - A `ScorerCache` represents the cross between a scorer and a corpus. Do not try to use a
   single cache across multiple scorers or corpora -- you'll get unexpected/invalid results.

Example:

```python
import pyterrier as pt
pt.init()
from pyterrier_caching import ScorerCache

# Setup
cached_scorer = ScorerCache('path/to/cache', MyExpensiveScorer())
dataset = pt.get_dataset('some-dataset') # e.g., 'irds:msmarco-passage'

# You need to build your cache before you can use it. There are several
# ways to do this:
if not cached_scorer.built():
    # Easiest:
    cached_scorer.build(dataset.get_corpus_iter())
    # If you already have an "npids" file to map the docnos to indexes, you can use:
    # >>> cached_scorer.build(docnos_file='path/to/docnos.npids')
    # This will be faster than iterating over the entire corpus, especially for
    # large datasets.

# Use the IndexerCache cache object just as you would an indexer
cached_pipeline = MyFirstStage() >> cached_scorer

cached_pipeline(dataset.get_topics())
# Will be faster when you run it a second time, since all values are cached
cached_pipeline(dataset.get_topics())

# Will only compute scores for docnos that were not returned by MyFirstStage()
another_cached_pipeline = AnotherFirstStage() >> cached_scorer
another_cached_pipeline(dataset.get_topics())
```

<details>
<summary>👁‍🗨 More Details</summary>

`ScorerCache` currently has one implementation, `Hdf5ScorerCache`, which is
set as the default. `Hdf5ScorerCache` saves scores in an HDF5 file.

</details>

## Extras

These components are not caching _per se_, but can be helpful when constructing a
caching pipeline.

`Lazy(...)` allows you to build a transformer object that is only initialized when
it is first executed. This can help avoid the expensive process of reading and loading
a model that may never be executed due to caching.

For example, this example uses `Lazy` with a `ScorerCache` to avoid loading `MyExpensiveTransformer`
unless it's actually needed:

```python
from pyterrier_caching import Lazy, ScorerCache

lazy_transformer = Lazy(lambda: MyExpensiveTransformer())
cache = ScorerCache('path/to/cache', lazy_transformer)
```
