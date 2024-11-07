from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pyterrier as pt
import shutil
import json
import struct
from npids import Lookup
from deprecated import deprecated
from pyterrier_caching import BuilderMode, artifact_builder, meta_file_compat, dbm_sqlite3
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
            if self.scorer is None:
                raise LookupError('values missing from cache, but no scorer provided')
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


class DbmSqlite3ScorerCache(pta.Artifact, pt.Transformer):
    artifact_type = 'scorer_cache'
    artifact_format = 'dbm.sqlite3'

    def __init__(self, path, scorer=None, *, group_by=None, key=None, verbose=False):
        super().__init__(path)
        meta_file_compat(path)
        self.scorer = scorer
        self.verbose = verbose
        self.meta = None
        if not (Path(self.path)/'pt_meta.json').exists():
            if group_by is None:
                group_by = 'query'
            if key is None:
                key = 'docno'
            with artifact_builder(self.path, BuilderMode.create, self.artifact_type, self.artifact_format) as builder:
                builder.metadata['group_by'] = group_by
                builder.metadata['key'] = key
        with (Path(self.path)/'pt_meta.json').open('rt') as fin:
            self.meta = json.load(fin)
        if group_by is not None:
            assert group_by == self.meta['group_by'], f'group_by={group_by!r} provided, but index created with group_by={self.meta["group_by"]!r}'
        if key is not None:
            assert key == self.meta['key'], f'key={key!r} provided, but index created with key={self.meta["key"]!r}'
        self.group_by = self.meta['group_by']
        self.key = self.meta['key']

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.columns(inp, includes=[self.group_by, self.key])

        results = []
        to_score = []
        to_score_idxs = {}
        scores = []

        # First pass: load what we can from caches
        for group_by_key, group in inp.groupby(self.group_by):
            group_by_hash = hashlib.sha256(group_by_key.encode('utf8')).hexdigest()
            cache_path = Path(self.path) / group_by_hash[:2] / f'{group_by_hash}.sqlite3'
            if not cache_path.parent.exists():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not cache_path.exists():
                with dbm_sqlite3.open(cache_path, flag='c'):
                    pass
            results.append(group)
            with dbm_sqlite3.open(cache_path, flag='r') as fin:
                for row in group.itertuples(index=False):
                    key = getattr(row, self.key).encode('utf8')
                    if key in fin:
                        scores.append(struct.unpack('<d', fin[key])[0])
                    else:
                        to_score.append(row)
                        to_score_idxs[group_by_hash, key] = len(scores)
                        scores.append(None)

        # Second pass: score the missing ones and add to cache
        if to_score:
            to_score = pd.DataFrame(to_score)
            if self.scorer is None:
                raise LookupError('values missing from cache, but no scorer provided')
            scored = self.scorer(to_score)
            for group_by_key, group in scored.groupby(self.group_by):
                group_by_hash = hashlib.sha256(group_by_key.encode('utf8')).hexdigest()
                cache_path = Path(self.path) / group_by_hash[:2] / f'{group_by_hash}.sqlite3'
                with dbm_sqlite3.open(cache_path, flag='w') as fout:
                    for row in group.itertuples(index=False):
                        key = getattr(row, self.key).encode('utf8')
                        scores[to_score_idxs[group_by_hash, key]] = row.score
                        fout[key] = struct.pack('<d', row.score)

        results = pd.concat(results, ignore_index=True).assign(score=scores)
        pt.model.add_ranks(results)
        if self.verbose:
            print(f"{self}: {len(inp)-len(to_score)} hit(s), {len(to_score)} miss(es)")
        return results

    def __repr__(self):
        return f'DbmSqlite3ScorerCache({str(self.path)!r}, {self.scorer!r}, group_by={self.group_by!r}, key={self.key!r})'

@deprecated(version='0.2.0', reason='ScorerCache will be switched from the dense `Hdf5ScorerCache` implementation to '
                                    'the sparse `DbmSqlite3ScorerCache` in a future version, which may break '
                                    'functionality that relies on it being a dense cache. Switch to DenseScorerCache '
                                    'instead.')
class DeprecatedHdf5ScorerCache(Hdf5ScorerCache):
    pass

# Default implementations
ScorerCache = DeprecatedHdf5ScorerCache
DenseScorerCache = Hdf5ScorerCache
SparseScorerCache = DbmSqlite3ScorerCache
