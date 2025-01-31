__version__ = '0.4.1'

from pyterrier_caching.util import Lazy, closing_memmap, meta_file_compat
from pyterrier_caching.indexer_cache import IndexerCache, Lz4PickleIndexerCache
from pyterrier_caching.scorer_cache import ScorerCache, DenseScorerCache, SparseScorerCache, Hdf5ScorerCache, Sqlite3ScorerCache
from pyterrier_caching.retriever_cache import RetrieverCache, DbmRetrieverCache
from pyterrier_caching.key_value_cache import KeyValueCache, Sqlite3KeyValueCache

__all__ = ["Lazy", "closing_memmap", "meta_file_compat", "IndexerCache", "Lz4PickleIndexerCache", "ScorerCache", "DenseScorerCache", "SparseScorerCache", "Hdf5ScorerCache", "Sqlite3ScorerCache", "RetrieverCache", "DbmRetrieverCache", 'KeyValueCache', 'Sqlite3KeyValueCache']
