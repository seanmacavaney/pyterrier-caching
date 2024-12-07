Caching Retriever Results
====================================

:class:`~pyterrier_caching.RetrieverCache` saves the retrieved results based on the fields
of each row. When the same row is encountered again, the value is read from the cache,
avoiding retrieving again.

**Example use case:** I want to test several different re-ranking models over the same
initial set of documents, and I want to save time by not re-running the queries each time.

You use a ``RetrieverCache`` in place of the retriever in a pipeline. It holds a reference to
the retriever so that it can retrieve results for queries that are missing from the cache.

.. warning::

   **Important Caveats**:
   
   - ``RetrieverCache`` saves scores based on **all** the input columns by default. Changes in
     any of the values will result in a cache miss, even if the column does not affect the
     retriever's output. You can specify a subset of columns using the ``on`` parameter.
   - DBM does not support concurrent reads/writes from multiple threads or processes. Keep only
     a single ``RetrieverCache`` pointing to a cache file location open at a time.
   - A ``RetrieverCache`` represents the cross between a retriever and a corpus. Do not try to use a
     single cache across multiple retrievers or corpora -- you'll get unexpected/invalid results.

Example:

.. code-block:: python

    import pyterrier as pt
    from pyterrier_caching import RetrieverCache

    # Setup
    cached_retriever = RetrieverCache('path/to/cache', MyRetriever())
    dataset = pt.get_dataset('some-dataset') # e.g., 'irds:msmarco-passage'

    # Use the RetrieverCache cache object just as you would a retriever
    cached_pipeline = cached_retriever >> MySecondStage()

    cached_pipeline(dataset.get_topics())
    # Will be faster when you run it a second time, since all values are cached
    cached_pipeline(dataset.get_topics())

API Documentation
--------------------------

.. autoclass:: pyterrier_caching.RetrieverCache
   :members:

.. autoclass:: pyterrier_caching.DbmRetrieverCache
   :members:
