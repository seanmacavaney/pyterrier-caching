from pathlib import Path
import hashlib
import pandas as pd
import pyterrier as pt
import shelve
import json
from pyterrier_caching import BuilderMode, artefact_builder


class ShelveRetrieverCache(pt.Transformer):
    artefact_type = 'retriever_cache'
    artefact_format = 'shelve'

    def __init__(self, path, retriever=None, on='query'):
        self.on = on
        self.path = Path(path)
        self.retriever = retriever
        self.meta = None
        self.file = None
        if not (Path(self.path)/'meta.json').exists():
            with artefact_builder(self.path, BuilderMode.create, self.artefact_type, self.artefact_format) as builder:
                with shelve.open(str(builder.path/'data.shelve'), 'c') as data:
                    pass # just create the data file

    def transform(self, inp):
        assert self.on in inp.columns
        self._ensure_built()
        any_updates = False
        results = []
        to_retrieve = []
        for key, group in inp.groupby(self.on):
            key_hash = self.on + '_' + hashlib.sha256(key.encode()).hexdigest()
            if key_hash in self.file:
                results.append(self.file[key_hash])
            else:
                to_retrieve.append(group)
        if to_retrieve:
            self.file.close()
            self.file = None
            retrieved_results = self.retriever(pd.concat(to_retrieve))
            results.append(retrieved_results)
            with shelve.open(str(self.path/'data.shelve'), 'w') as file:
                for key, group in retrieved_results.groupby(self.on):
                    key_hash = self.on + '_' + hashlib.sha256(key.encode()).hexdigest()
                    file[key_hash] = group
        return pd.concat(results, ignore_index=True)

    def _ensure_built(self):
        if self.file is None:
            self.file = shelve.open(str(self.path/'data.shelve'), 'r')
        if self.meta is None:
            with (self.path/'meta.json').open('rt') as fin:
                self.meta = json.load(fin)
            assert self.meta['type'] == self.artefact_type
            assert self.meta['format'] == self.artefact_format


# Default implementation of RetrieverCache: ShelveRetrieverCache
RetrieverCache = ShelveRetrieverCache
