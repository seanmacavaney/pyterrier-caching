from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pyterrier as pt
import shutil
import json
from npids import Lookup
from pyterrier_caching import BuilderMode, artifact_builder, meta_file_compat
import pyterrier_alpha as pta


class Hdf5ScorerCache(pta.Artifact, pt.Transformer):
    artifact_type = 'scorer_cache'
    artifact_format = 'hdf5'

    def __init__(self, path, scorer=None, verbose=False):
        super().__init__(path)
        meta_file_compat(path)
        self.mode = 'r'
        self.scorer = scorer
        self.verbose = verbose
        self.meta = None
        self.file = None
        self.docnos = None
        self.dataset_cache = {}

    def transform(self, inp):
        return self.cached_scorer()(inp)

    def built(self) -> bool:
        return (Path(self.path)/'pt_meta.json').exists()

    def build(self, corpus_iter=None, docnos_file=None):
        assert not self.built(), "this cache is alrady built"
        assert corpus_iter is not None or docnos_file is not None
        import h5py
        with artifact_builder(self.path, BuilderMode.create, self.artifact_type, self.artifact_format) as builder:
            with h5py.File(str(builder.path/'data.h5'), 'a'):
                pass # just create the data file
            if docnos_file:
                shutil.copy(docnos_file, builder.path/'docnos.npids')
                builder.metadata['doc_count'] = len(Lookup(builder.path/'docnos.npids'))
            else:
                builder.metadata['doc_count'] = 0
                with Lookup.builder(builder.path/'docnos.npids') as docno_lookup:
                    for record in corpus_iter:
                        docno_lookup.add(record['docno'])
                        builder.metadata['doc_count'] += 1

    def _ensure_built(self):
        import h5py
        assert self.built(), "you must .build(...) this cache before it can be used"
        if self.file is None:
            self.file = h5py.File(self.path/'data.h5', self.mode)
        if self.meta is None:
            with (self.path/'pt_meta.json').open('rt') as fin:
                self.meta = json.load(fin)
        if self.docnos is None:
            self.docnos = Lookup(self.path/'docnos.npids')

    def _ensure_write_mode(self):
        if self.mode == 'r':
            assert self.scorer is not None, "missing value in cache; scorer is required"
            import h5py
            self.mode = 'a'
            self.file.close()
            self.file = h5py.File(self.path/'data.h5', self.mode)
            self.dataset_cache = {} # file changed, need to reset the cache

    def _get_dataset(self, qid):
        if qid not in self.dataset_cache:
            if qid not in self.file:
                self._ensure_write_mode()
                # TODO: setting chunks=(4096,) --- or some other value? --- might help
                # reduce the file size and/or speed up writes? Investigate more...
                self.file.create_dataset(qid, shape=(self.corpus_count(),), dtype=np.float32, fillvalue=float('nan'))
            self.dataset_cache[qid] = self.file[qid]
        return self.dataset_cache[qid]

    def corpus_count(self):
        self._ensure_built()
        return self.meta['doc_count']

    def __repr__(self):
        return f'Hdf5ScorerCache({repr(str(self.path))}, {self.scorer})'

    def cached_scorer(self) -> pt.Transformer:
        return Hdf5ScorerCacheScorer(self)

    def cached_retriever(self, num_results: int = 1000) -> pt.Transformer:
        return Hdf5ScorerCacheRetriever(self, num_results)


class Hdf5ScorerCacheScorer(pt.Transformer):
    def __init__(self, cache: Hdf5ScorerCache):
        self.cache = cache

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        self.cache._ensure_built()
        results = []
        misses = 0
        for query, group in inp.groupby('query'):
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            ds = self.cache._get_dataset(query_hash)
            dids = self.cache.docnos.inv[np.array(group.docno)]
            dids_sorted, undo_did_sort = np.unique(dids, return_inverse=True)
            scores = ds[dids_sorted][undo_did_sort]
            to_score = group.loc[group.docno[np.isnan(scores)].index]
            misses += len(to_score)
            if len(to_score) > 0:
                self.cache._ensure_write_mode()
                ds = self.cache._get_dataset(query_hash)
                new_scores = self.cache.scorer(to_score)
                dids = self.cache.docnos.inv[np.array(new_scores.docno)]
                dids_sorted, dids_sorted_idx = np.unique(dids, return_index=True)
                ds[dids_sorted] = new_scores.score.iloc[dids_sorted_idx]
                dids = self.cache.docnos.inv[np.array(group.docno)]
                dids_sorted, undo_did_sort = np.unique(dids, return_inverse=True)
                scores = ds[dids_sorted][undo_did_sort]
            results.append(group.assign(score=scores))
        results = pd.concat(results, ignore_index=True)
        pt.model.add_ranks(results)
        if self.cache.verbose:
            print(f"{self}: {len(inp)-misses} hit(s), {misses} miss(es)")
        return results


class Hdf5ScorerCacheRetriever(pt.Transformer):
    def __init__(self, cache: Hdf5ScorerCache, num_results: int = 1000):
        self.cache = cache
        self.num_results = num_results

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        self.cache._ensure_built()
        pta.validate.query_frame(inp, extra_columns=['query'])
        inp = inp.reset_index(drop=True)
        builder = pta.DataFrameBuilder(['_index', 'docno', 'score', 'rank'])
        for i, query in enumerate(inp['query']):
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            ds = self.cache._get_dataset(query_hash)[:]
            nans = np.isnan(ds)
            if nans.any():
                raise RuntimeError(f'retriever only works if corpus is scored completely; '
                                   f'{nans.sum()} uncached documents found for query {query!r}.')
            k = min(len(ds), self.num_results)
            docids = ds.argpartition(-k)[-k:]
            scores = ds[docids]
            idxs = scores.argsort()[::-1]
            builder.extend({
                '_index': i,
                'docno': self.cache.docnos.fwd[docids[idxs]],
                'score': scores[idxs],
                'rank': np.arange(scores.shape[0]),
            })
        return builder.to_df(merge_on_index=inp)


# Default implementation of ScorerCache: Hdf5ScorerCache
ScorerCache = Hdf5ScorerCache
