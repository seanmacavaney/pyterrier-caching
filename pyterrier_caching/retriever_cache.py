from typing import Optional, Union
from pathlib import Path
import hashlib
import lz4.frame
import pandas as pd
import pyterrier as pt
import pickle
import json
import dbm.dumb
import pyterrier_alpha as pta
from pyterrier_caching import meta_file_compat


class DbmRetrieverCache(pta.Artifact, pt.Transformer):
    """A :class:`~pyterrier_caching.RetrieverCache` that stores retrieved results in ``dbm.dumb`` database files."""
    ARTIFACT_TYPE = 'retriever_cache'
    ARTIFACT_FORMAT = 'dbm.dumb'

    def __init__(self,
                 path: Union[str, Path],
                 retriever: Optional[pt.Transformer] = None,
                 on: Optional[str] = None,
                 verbose: bool = False):
        """
        Args:
            path: The path to the cache.
            retriever: The retriever that is cached.
            on: The column(s) to use as the key for the cache. If None, all columns will be used.
            verbose: If True, print progress information.
        """
        super().__init__(path)
        meta_file_compat(path)
        self.on = on
        self.retriever = retriever
        self.verbose = verbose
        self.meta = None
        self.file = None
        self.file_name = None
        if not (Path(self.path)/'pt_meta.json').exists():
            with pta.ArtifactBuilder(self):
                pass # just create the artifact

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        if self.on is not None:
            if isinstance(self.on, str):
                assert self.on in inp.columns
                on = [self.on]
            else:
                assert all(o in inp.columns for o in self.on)
                on = list(self.on)
        else:
            on = inp.columns
        on = tuple(sorted(on))

        self._ensure_built(on)
        results = []
        to_retrieve = []
        for i in range(len(inp)):
            row = inp.iloc[i]
            key = tuple(row[o] for o in on)
            key_hash = hashlib.sha256(pickle.dumps(key)).digest()
            if key_hash in self.file:
                stored_data = pickle.loads(lz4.frame.decompress(self.file[key_hash]))
                results.append(pd.DataFrame(stored_data))
            else:
                to_retrieve.append((i, key_hash))
        if to_retrieve:
            self.file.close()
            self.file = None
            with dbm.dumb.open(self.file_name, 'w') as file:
                self.file_name = None
                it = to_retrieve
                if self.verbose:
                    it = pt.tqdm(it, unit='q')
                for i, key_hash in it:
                    retrieved_results = self.retriever(inp.iloc[i:i+1])
                    results.append(retrieved_results)
                    stored_data = {c: retrieved_results[c].values for c in retrieved_results.columns}
                    file[key_hash] = lz4.frame.compress(pickle.dumps(stored_data))
        if self.verbose:
            print(f'{self}: {len(inp)-len(to_retrieve)} hit(s), {len(to_retrieve)} miss(es)')
        return pd.concat(results, ignore_index=True)

    def _ensure_built(self, on):
        on_hash = hashlib.sha256(pickle.dumps(on)).hexdigest()
        fname = str(self.path/f'{on_hash}.dbm')
        if self.file_name is not None and self.file_name != fname:
            self.file.close()
            self.file = None
            self.file_name = None
        if self.file is None:
            self.file = dbm.dumb.open(fname, 'c')
            self.file_name = fname
        if self.meta is None:
            with (self.path/'pt_meta.json').open('rt') as fin:
                self.meta = json.load(fin)
        assert self.meta['type'] == self.ARTIFACT_TYPE
        assert self.meta['format'] == self.ARTIFACT_FORMAT

    def __repr__(self):
        return f'DbmRetrieverCache({repr(str(self.path))}, {self.retriever})'


# Default implementation of RetrieverCache: DbmRetrieverCache
RetrieverCache = DbmRetrieverCache
