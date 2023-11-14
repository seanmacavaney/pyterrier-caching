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


class Hdf5ScorerCache(pt.Transformer):
    artefact_type = 'scorer_cache'
    artefact_format = 'hdf5'

    def __init__(self, path, scorer=None):
        self.mode = 'r'
        self.path = Path(path)
        self.scorer = scorer
        self.meta = None
        self.file = None
        self.docnos = None
        self.dataset_cache = {}

    def transform(self, inp):
        self._ensure_built()
        results = []
        for query, group in inp.groupby('query'):
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            ds = self._get_dataset(query_hash)
            dids = self.docnos.inv[np.array(group.docno)]
            dids_sorted, undo_did_sort = np.unique(dids, return_inverse=True)
            scores = ds[dids_sorted][undo_did_sort]
            to_score = group.loc[group.docno[np.isnan(scores)].index]
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
        if self.file is None:
            self.file = h5py.File(self.path/'data.h5', self.mode)
        if self.meta is None:
            with (self.path/'meta.json').open('rt') as fin:
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

# Default implementation of ScorerCache: Hdf5ScorerCache
ScorerCache = Hdf5ScorerCache
