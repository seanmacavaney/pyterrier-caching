Caching Scorer / Re-Ranker Results
====================================

Scorers (Re-Rankers) are a common type of :class:`~pyterrier.Transformer` that re-order a set
of results with a new scoring function. Scorers are often neural cross-encoders, e.g.
:class:`pyterrier_dr.ElectraScorer`.

Scorers can be expensive to execute, so it can be helpful to cache the results throughout the
course of experimentation. For example, you may want to test how a neural relevance model performs
over several first-stage retrieval models that give back many of the same results.

:class:`~pyterrier_caching.ScorerCache` saves the ``score`` based on ``query`` and ``docno`` [#names]_.
When the same ``query``-``docno`` combination is encountered again, the score is read from the cache,
avoiding re-computation.

You use a ``ScorerCache`` in place of the scorer in a pipeline. It holds a reference to
the scorer so that it can compute values that are missing from the cache.

.. warning::
   **Important Caveats**:

   - ``ScorerCache`` saves scores based on **only** the value of the ``query`` and ``docno`` [#names]_.
     All other information is ignored (e.g., the text of the document). Note that this strategy 
     makes it suitable only when each score only depends on the ``query`` and ``docno`` of a single 
     record (e.g., Mono-style models) and not cases that perform pairwise or listwise scoring 
     (e.g, Duo-style models).
   - ``ScorerCache`` only stores the result of the ``score`` column. All other outputs of the scorer
     are discarded. (Rank is also given in the output, but it is calculated by cache, not the scorer.)
   - Scores are saved as ``float64`` values. Other values will be attempted to be cast up/down to `float64`.
   - A ``ScorerCache`` represents the cross between a scorer and a corpus. Do not try to 
     use a single cache across multiple scorers or corpora -- you'll get unexpected/invalid 
     results.

Example:

.. code-block:: python
   :caption: Caching MonoElectra results using :class:`~pyterrier_caching.ScorerCache`

   import pyterrier as pt
   from pyterrier_caching import ScorerCache
   from pyterrier_dr import ElectraScorer
   from pyterrier_pisa import PisaIndex

   # Setup
   dataset = pt.get_dataset('irds:msmarco-passage/dev/small')
   index = PisaIndex.from_hf('macavaney/msmarco-passage.pisa')
   scorer = dataset.text_loader() >> ElectraScorer()
   cached_scorer = ScorerCache('electra.cache', scorer)

   # Use the ScorerCache cache object just as you would a scorer
   cached_pipeline = index.bm25() >> cached_scorer
   cached_pipeline(dataset.get_topics())

   # Will be faster when you run it a second time, since all values are cached
   cached_pipeline(dataset.get_topics())

   # Will only compute scores for docnos that were not returned by bm25
   another_cached_pipeline = index.qld() >> cached_scorer
   another_cached_pipeline(dataset.get_topics())

Advanced
--------------------------

Caching Learning-to-rank Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can cache learning-to-rank features by setting ``value="features"`` and ``pickle=True`` when constructing
the cache.

Example:

.. code-block:: python
   :caption: Cache learning-to-rank features with ``ScorerCache``

   from pyterrier_caching import ScorerCache
   feature_extractor = ... # a transformer that extracts features based on query and docno
   cache = ScorerCache('mycache', feature_extractor, value="features", pickle=True)


API Documentation
--------------------------

.. autoclass:: pyterrier_caching.ScorerCache
   :members:

.. autoclass:: pyterrier_caching.SparseScorerCache
   :members:

.. autoclass:: pyterrier_caching.Sqlite3ScorerCache
   :members:

.. autoclass:: pyterrier_caching.DenseScorerCache
   :members:

.. autoclass:: pyterrier_caching.Hdf5ScorerCache
   :members:

--------------------------

.. [#names] These fields can be configured with ``group`` (query), ``key`` (docno), and ``value`` (score) settings.
