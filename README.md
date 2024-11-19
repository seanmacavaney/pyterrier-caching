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

Concrete Examples:
 - [examples/indexer_splade.py](examples/indexer_splade.py)

<details>
<summary>👁‍ More Details</summary>

`IndexerCache` currently has one implementation, `Lz4PickleIndexerCache`, which is
set as the default. `Lz4PickleIndexerCache` saves the sequence as a sequence of
LZ4-compressed pickled dicts in the file: `data.pkl.lz4`. Byte-level offsets for each
document are stored as a numpy-compatible float64 array in `offsets.np`. If the `docno`
column is present, an [`npids`]() structure is also stored, facilitating reverse-lookups
of documents by their docno.

</details>


## Caching Results from a Scorer

`SparseScorerCache` saves the `score` based on `query` and `docno`. When the same
`query`-`docno` combination is encountered again, the value is read from the cache,
avoiding re-computation.

**Example use case:** I want to test a neural relevance model over several first-stage
retrieval models, but they bring back many of the same documents, so I don't want to
re-compute the scores each time.

You use a `SparseScorerCache` in place of the scorer in a pipeline. It holds a reference to
the scorer so that it can compute values that are missing from the cache.

**⚠️ Important Caveats**:
 - `SparseScorerCache` saves scores based on **only** the value of the `query` and `docno`. All
   other information is ignored (e.g., the text of the document). Note that this strategy
   makes it suitable only when each score only depends on the `query` and `docno` of a single
   record (e.g., Mono-style models) and not cases that perform pairwise or listwise scoring
   (e.g, Duo-style models).
 - `SparseScorerCache` only stores the result of the `score` column. All other outputs of the
   scorer will be discarded. (Rank is also outputed, but is caculated by `SparseScorerCache`,
   not the scorer.)
 - Scores are saved as `float64` values. Other values will attempted to be cast up/down
   to float64.
 - A `SparseScorerCache` represents the cross between a scorer and a corpus. Do not try to use a
   single cache across multiple scorers or corpora -- you'll get unexpected/invalid results.

Example:

```python
import pyterrier as pt
from pyterrier_caching import SparseScorerCache

# Setup
cached_scorer = SparseScorerCache('path/to/cache', MyExpensiveScorer())
dataset = pt.get_dataset('some-dataset') # e.g., 'irds:msmarco-passage'

# Use the SparseScorerCache cache object just as you would a scorer
cached_pipeline = MyFirstStage() >> cached_scorer

cached_pipeline(dataset.get_topics())
# Will be faster when you run it a second time, since all values are cached
cached_pipeline(dataset.get_topics())

# Will only compute scores for docnos that were not returned by MyFirstStage()
another_cached_pipeline = AnotherFirstStage() >> cached_scorer
another_cached_pipeline(dataset.get_topics())
```

Concrete Examples:
 - [examples/scorer_monot5.py](examples/scorer_monot5.py)

<details>
<summary>👁‍ More Details</summary>

`SparseScorerCache` is suitable only when a relatively small proportion of a corpus is cached.
If the proportion grows larger, consider using `DenseScorerCache` instead, which stores results
using HDF5. When using a `DenseScorerCache`, you need an initial "build" step, which constructs
a mapping of the docnos to their index in the lookup vector:

```python
import pyterrier as pt
from pyterrier_caching import SparseScorerCache

# Setup
cached_scorer = DenseScorerCache('path/to/cache', MyExpensiveScorer())
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

# Use the DenseScorerCache cache object just as you would a scorer
cached_pipeline = MyFirstStage() >> cached_scorer

cached_pipeline(dataset.get_topics())
# Will be faster when you run it a second time, since all values are cached
cached_pipeline(dataset.get_topics())

# Will only compute scores for docnos that were not returned by MyFirstStage()
another_cached_pipeline = AnotherFirstStage() >> cached_scorer
another_cached_pipeline(dataset.get_topics())
```

</details>


## Caching Results from a Retriever

`RetrieverCache` saves the retrieved results based on the fields of each row. When the
same row is encountered again, the value is read from the cache, avoiding retrieving again.

**Example use case:** I want to test several different re-ranking models over the same
initial set of documents, and I want to save time by not re-running the queries each time.

You use a `RetrieverCache` in place of the retriever in a pipeline. It holds a reference to
the retriever so that it can retrieve results for queries that are missing from the cache.

**⚠️ Important Caveats**:
 - `RetrieverCache` saves scores based on **all** the input columns by default. Changes in
   any of the values will result in a cache miss, even if the column does not affect the
   retriever's output. You can specify a subset of columns using the `on` parameter.
 - DBM does not support concurrent reads/writes from multiple threads or processes. Keep only
   a single `RetrieverCache` pointing to a cache file location open at a time.
 - A `ScorerCache` represents the cross between a retriever and a corpus. Do not try to use a
   single cache across multiple retrievers or corpora -- you'll get unexpected/invalid results.

Example:

```python
import pyterrier as pt
pt.init()
from pyterrier_caching import RetrieverCache

# Setup
cached_retriever = RetrieverCache('path/to/cache', MyRetriever())
dataset = pt.get_dataset('some-dataset') # e.g., 'irds:msmarco-passage'

# Use the RetrieverCache cache object just as you would a retriever
cached_pipeline = cached_retriever >> MySecondStage()

cached_pipeline(dataset.get_topics())
# Will be faster when you run it a second time, since all values are cached
cached_pipeline(dataset.get_topics())
```

Concrete Examples:
 - [examples/retriever_bm25.py](examples/retriever_bm25.py)

<details>
<summary>👁‍ More Details</summary>

`RetrieverCache` currently has one implementation, `DbmScorerCache`, which is
set as the default. `DbmScorerCache` saves results as a
[`dbm`](https://docs.python.org/3/library/dbm.html) file.

</details>

## Extras

You load caches from HuggingFace Hub and push caches to HuggingFace Hub using `.from_hf('id')`
and  `.to_hf('id')`. Example:

```python
from pyterrier_caching import ScorerCache
cache = ScorerCache.from_hf('macavaney/msmarco-passage.monot5-base.cache')
cache.to_hf('username/dataset')
```

The following components are not caching _per se_, but can be helpful when constructing a
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
