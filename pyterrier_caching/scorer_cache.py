from typing import Optional, List, Iterator, Dict, Any, Union
from pathlib import Path
from contextlib import ExitStack
import struct
import hashlib
import pickle
import lz4.frame
import numpy as np
import pandas as pd
import pyterrier as pt
import shutil
import json
from npids import Lookup
from pyterrier_caching import BuilderMode, artefact_builder


class Hdf5DenseScorerCache(pt.Transformer):
    artefact_type = 'scorer_cache'
    artefact_format = 'hdf5'

    def __init__(self, path, scorer=None, verbose=False):
        self.mode = 'r'
        self.path = Path(path)
        self.scorer = scorer
        self.verbose = verbose
        self.meta = None
        self.file = None
        self.docnos = None
        self.dataset_cache = {}

    def transform(self, inp):
        self._ensure_built()
        results = []
        misses = 0
        for query, group in inp.groupby('query'):
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            ds = self._get_dataset(query_hash)
            dids = self.docnos.inv[np.array(group.docno)]
            dids_sorted, undo_did_sort = np.unique(dids, return_inverse=True)
            scores = ds[dids_sorted][undo_did_sort]
            to_score = group.loc[group.docno[np.isnan(scores)].index]
            misses += len(to_score)
            if len(to_score) > 0:
                self._ensure_write_mode()
                ds = self._get_dataset(query_hash)
                new_scores = self.scorer(to_score)
                dids = self.docnos.inv[np.array(new_scores.docno)]
                dids_sorted, dids_sorted_idx = np.unique(dids, return_index=True)
                ds[dids_sorted] = new_scores.score.iloc[dids_sorted_idx]
                dids = self.docnos.inv[np.array(group.docno)]
                dids_sorted, undo_did_sort = np.unique(dids, return_inverse=True)
                scores = ds[dids_sorted][undo_did_sort]
            results.append(group.assign(score=scores))
        results = pd.concat(results, ignore_index=True)
        pt.model.add_ranks(results)
        if self.verbose:
            print(f"{self}: {len(inp)-misses} hit(s), {misses} miss(es)")
        return results

    def built(self) -> bool:
        return (Path(self.path)/'meta.json').exists()

    def build(self, corpus_iter=None, docnos_file=None):
        assert not self.built(), "this cache is alrady built"
        assert corpus_iter is not None or docnos_file is not None
        import h5py
        with artefact_builder(self.path, BuilderMode.create, self.artefact_type, self.artefact_format) as builder:
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
        if self.meta is None:
            with (self.path/'meta.json').open('rt') as fin:
                meta = json.load(fin)
            assert meta['type'] == self.artefact_type
            assert meta['format'] == self.artefact_format
            self.meta = meta
        if self.file is None:
            self.file = h5py.File(self.path/'data.h5', self.mode)
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
        return f'Hdf5DenseScorerCache({repr(str(self.path))}, {self.scorer})'


class Hdf5SparseScorerCache(pt.Transformer):
    artefact_type = 'scorer_cache'
    artefact_format = 'hdf5_sparse'

    def __init__(self, path, scorer=None, verbose=False):
        self.mode = 'r'
        self.path = Path(path)
        self.scorer = scorer
        self.verbose = verbose
        self.meta = None
        self.file = None
        self.docnos = None
        self.dataset_cache = {}
        self.dataset_vec_cache = {}

    def transform(self, inp):
        self._ensure_built()
        results = []
        misses = 0
        for query, group in inp.groupby('query'):
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            ds_idxs, ds_vals = self._get_dataset_vector(query_hash)
            dids = self.docnos.inv[np.array(group.docno)]
            dids_sorted, undo_did_sort = np.unique(dids, return_inverse=True)
            search_idxs = np.searchsorted(ds_idxs, dids_sorted)
            search_idxs_sorted, undo_search_idxs_sorted = np.unique(search_idxs, return_inverse=True)
            miss_mask = ds_idxs[search_idxs_sorted][undo_search_idxs_sorted] != dids_sorted
            if miss_mask.any():
                misses += miss_mask.sum()
                to_score = group.loc[group.docno[miss_mask[undo_did_sort]].index]
                self._ensure_write_mode()
                ds_idxs, ds_vals = self._get_dataset(query_hash)
                new_scores = self.scorer(to_score)
                new_dids = self.docnos.inv[np.array(new_scores.docno)]
                new_dids_sorted, new_dids_sorted_idx = np.unique(new_dids, return_index=True)
                new_all_idxs = np.concatenate([ds_idxs[:], new_dids_sorted])
                new_all_vals = np.concatenate([ds_vals[:], new_scores.score.iloc[new_dids_sorted_idx]])
                sort_idxs = np.argsort(new_all_idxs)
                ds_idxs.resize(new_all_idxs.shape)
                ds_vals.resize(new_all_idxs.shape)
                ds_idxs[:] = new_all_idxs[sort_idxs]
                ds_vals[:] = new_all_vals[sort_idxs]
                ds_idxs, ds_vals = ds_idxs[:], ds_vals[:]
                self.dataset_vec_cache[query_hash] = (ds_idxs, ds_vals)
                search_idxs = np.searchsorted(ds_idxs, dids_sorted)
                search_idxs_sorted, undo_search_idxs_sorted = np.unique(search_idxs, return_inverse=True)
                assert (ds_idxs[search_idxs_sorted][undo_search_idxs_sorted] == dids_sorted).all()
            results.append(group.assign(score=ds_vals[search_idxs][undo_did_sort]))
        results = pd.concat(results, ignore_index=True)
        pt.model.add_ranks(results)
        if self.verbose:
            print(f"{self}: {len(inp)-misses} hit(s), {misses} miss(es)")
        return results

    def built(self) -> bool:
        return (Path(self.path)/'meta.json').exists()

    def build(self, corpus_iter=None, docnos_file=None):
        assert not self.built(), "this cache is alrady built"
        assert corpus_iter is not None or docnos_file is not None
        import h5py
        with artefact_builder(self.path, BuilderMode.create, self.artefact_type, self.artefact_format) as builder:
            with h5py.File(str(builder.path/'sparse_data.h5'), 'a'):
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
        if self.meta is None:
            with (self.path/'meta.json').open('rt') as fin:
                meta = json.load(fin)
            assert meta['type'] == self.artefact_type
            assert meta['format'] == self.artefact_format
            self.meta = meta
        if self.file is None:
            self.file = h5py.File(self.path/'sparse_data.h5', self.mode)
        if self.docnos is None:
            self.docnos = Lookup(self.path/'docnos.npids')

    def _ensure_write_mode(self):
        if self.mode == 'r':
            assert self.scorer is not None, "missing value in cache; scorer is required"
            import h5py
            self.mode = 'a'
            self.file.close()
            self.file = h5py.File(self.path/'sparse_data.h5', self.mode)
            self.dataset_cache = {} # file changed, need to reset the cache

    def _get_dataset(self, qid):
        if qid not in self.dataset_cache:
            if f'{qid}_idxs' not in self.file:
                self._ensure_write_mode()
                self.file.create_dataset(f'{qid}_idxs', fillvalue=np.iinfo(np.int64).max, dtype=np.int64, shape=(1,), maxshape=(None,))
                self.file.create_dataset(f'{qid}_vals', fillvalue=np.nan, dtype=np.float32, shape=(1,), maxshape=(None,))
            self.dataset_cache[qid] = (self.file[f'{qid}_idxs'], self.file[f'{qid}_vals'])
        return self.dataset_cache[qid]

    def _get_dataset_vector(self, qid):
        if qid not in self.dataset_vec_cache:
            if f'{qid}_idxs' not in self.file:
                self._ensure_write_mode()
                self.file.create_dataset(f'{qid}_idxs', fillvalue=np.iinfo(np.int64).max, dtype=np.int64, shape=(1,), maxshape=(None,))
                self.file.create_dataset(f'{qid}_vals', fillvalue=np.nan, dtype=np.float32, shape=(1,), maxshape=(None,))
            self.dataset_vec_cache[qid] = (self.file[f'{qid}_idxs'][:], self.file[f'{qid}_vals'][:])
        return self.dataset_vec_cache[qid]

    def corpus_count(self):
        self._ensure_built()
        return self.meta['doc_count']

    def __repr__(self):
        return f'Hdf5SparseScorerCache({repr(str(self.path))}, {self.scorer})'


# Default implementation of ScorerCache: Hdf5SparseScorerCache
Hdf5ScorerCache = Hdf5DenseScorerCache # backward compat
ScorerCache = Hdf5DenseScorerCache
DenseScorerCache = Hdf5DenseScorerCache
SparseScorerCache = Hdf5SparseScorerCache
