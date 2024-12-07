Caching Indexing Pipeline Results
====================================

:class:`~pyterrier_caching.IndexerCache` saves the sequence of documents encountered
in an indexing pipeline. It allows you to repeat that sequence without needing to
re-execute the computations up to that point.

**Example use case:** I want to test how different retrieval engines perform over 
`learned sparse representations <https://arxiv.org/abs/2303.13416>`_, but I don't want to 
re-compute the representations each time.

You use an ``IndexerCache`` the same way you would use an indexer:
as the last component of a pipeline. Rather than building an index of the data, the ``IndexerCache``
will save your results to a file on disk. This file can be re-read by iterating over the cache
object with ``iter(cache)``.

Example:

.. code-block:: python
    :caption: Caching the results of an expensive transformer using :class:`~pyterrier_caching.IndexerCache`

    import pyterrier as pt
    from pyterrier_caching import IndexerCache

    # Setup
    cache = IndexerCache('path/to/cache')
    dataset = pt.get_dataset('irds:msmarco-passage')

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

``IndexerCache`` provides a variety of other functionality over the cached results. See the API
documentation below for more details.


API Documentation
--------------------------

.. autoclass:: pyterrier_caching.IndexerCache
   :members:

.. autoclass:: pyterrier_caching.Lz4PickleIndexerCache
   :members:
   :special-members: __len__, __iter__, __getitem__
