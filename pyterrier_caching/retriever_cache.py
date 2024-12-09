from typing import Optional, Union
from pathlib import Path
from warnings import warn
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
                on = [self.on]
            else:
                on = list(self.on)
            pta.validate.columns(inp, includes=on)
        else:
            on = list(inp.columns)
        on = tuple(sorted(on))

        self._ensure_built(on)
        results = []
        to_retrieve = []
        to_retrieve_hashes = []

        # Step 1: Check Cache
        for i in range(len(inp)):
            row = inp.iloc[i]
            key = tuple(row[o] for o in on)
            key_hash = hashlib.sha256(pickle.dumps(key)).digest()
            if key_hash in self.file:
                stored_data = pickle.loads(lz4.frame.decompress(self.file[key_hash]))
                results.append(pd.DataFrame(stored_data))
            else:
                to_retrieve.append(i)
                to_retrieve_hashes.append(key_hash)

        # Step 2: Retrieve and save missing results
        if to_retrieve:
            self.file.close()
            self.file = None
            with dbm.dumb.open(self.file_name, 'w') as file:
                self.file_name = None
                out_cols = pta.inspect.transformer_outputs(self.retriever, list(inp.columns), strict=False)
                if out_cols is not None and all(o in out_cols for o in on):
                    one_at_a_time = False
                    retrieve_phases = [to_retrieve]
                else:
                    warn("Running retriever one query at a time because retriever's outputs could not be determined or "
                         f"the outputs do not contain the cache key: {on}")
                    one_at_a_time = True
                    retrieve_phases = [[idx] for idx in to_retrieve]
                for i, idxs in enumerate(retrieve_phases):
                    retrieved_results = self.retriever(inp.iloc[idxs])
                    retrieved_results.reset_index(drop=True, inplace=True)
                    results.append(retrieved_results)
                    if one_at_a_time:
                        hash_groups = [(to_retrieve_hashes[i], retrieved_results)]
                    else:
                        keys = retrieved_results[list(on)].itertuples(index=False)
                        key_hashes = [hashlib.sha256(pickle.dumps(tuple(key))).digest() for key in keys]
                        hash_groups = retrieved_results.groupby(key_hashes)
                    for key_hash, group in hash_groups:
                        if isinstance(key_hash, tuple):
                            assert len(key_hash) == 1
                            key_hash = key_hash[0]
                        stored_data = {c: group[c].values for c in group.columns}
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
