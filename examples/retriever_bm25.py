from time import time
import pyterrier as pt
pt.init()
from pyterrier_caching import RetrieverCache

# Setup
bm25 = pt.BatchRetrieve.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel='BM25')
cached_retriever = RetrieverCache('bm25.cache', bm25, verbose=True)
dataset = pt.get_dataset('irds:msmarco-passage/dev/small')


t0 = time()
cached_retriever(dataset.get_topics())
t1 = time()
print(f'first invocation: {t1-t0:.3f}s')

t0 = time()
cached_retriever(dataset.get_topics())
t1 = time()
print(f'second invocation: {t1-t0:.3f}s')
