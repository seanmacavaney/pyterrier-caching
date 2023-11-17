# pip install -q git+https://github.com/naver/splade.git git+https://github.com/cmacdonald/pyt_splade.git
from time import time
import pyterrier as pt
pt.init()
from pyterrier_caching import IndexerCache
from pyt_splade import SpladeFactory

cache = IndexerCache('splade.cache')
if not cache.built():
    dataset = pt.get_dataset('irds:antique')
    cache_pipeline = SpladeFactory().indexing() >> cache
    t0 = time()
    cache_pipeline.index(dataset.get_corpus_iter())
    t1 = time()
    print(f'time to cache: {t1-t0:.3f}s')

indexer = pt.IterDictIndexer('./splade.terrier', pretokenised=True)
t0 = time()
indexer.index(iter(cache))
t1 = time()
print(f'time to index from cache: {t1-t0:.3f}s')
