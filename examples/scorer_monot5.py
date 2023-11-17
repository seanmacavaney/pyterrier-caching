from time import time
import pyterrier as pt
pt.init()
from pyterrier_caching import ScorerCache
from pyterrier_t5 import MonoT5ReRanker

# Setup
dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
bm25 = pt.BatchRetrieve.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel='BM25')
monot5 = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker()
cached_monot5 = ScorerCache('monot5.cache', monot5, verbose=True)
if not cached_monot5.built():
    cached_monot5.build(dataset.get_corpus_iter())
pipeline = bm25 >> cached_monot5


t0 = time()
pipeline(dataset.get_topics())
t1 = time()
print(f'first invocation: {t1-t0:.3f}s')

t0 = time()
pipeline(dataset.get_topics())
t1 = time()
print(f'second invocation: {t1-t0:.3f}s')
