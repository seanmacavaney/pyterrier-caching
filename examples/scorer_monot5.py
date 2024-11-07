from time import time
import pyterrier as pt
from pyterrier_caching import SparseScorerCache
from pyterrier_t5 import MonoT5ReRanker

# Setup
dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
bm25 = pt.terrier.Retriever.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel='BM25')
monot5 = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker()
cached_monot5 = SparseScorerCache('monot5.cache', monot5, verbose=True)
inp = bm25(dataset.get_topics())
pipeline = cached_monot5


t0 = time()
pipeline(inp)
t1 = time()
print(f'first invocation: {t1-t0:.3f}s')

t0 = time()
pipeline(inp)
t1 = time()
print(f'second invocation: {t1-t0:.3f}s')
