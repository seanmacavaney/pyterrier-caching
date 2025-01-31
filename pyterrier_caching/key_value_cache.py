from typing import Union, List, Tuple, Optional
from pathlib import Path
import pickle
import pandas as pd
import pyterrier as pt
import json
import sqlite3
from collections import defaultdict
from contextlib import closing, contextmanager
import pyterrier_alpha as pta

PICKLE_PROTOCOL = 4

class Sqlite3KeyValueCache(pta.Artifact, pt.Transformer):
    """A cache for storing and retrieving scores for documents, backed by a SQLite3 database.

    This is a *sparse* scorer cache, meaning that space is only allocated for documents that have been scored.
    If a large proportion of the corpus is expected to be scored, a dense cache (e.g., :class:`~pyterrier_caching.Hdf5ScorerCache`)
    may be more appropriate.
    """

    ARTIFACT_TYPE = 'key_value_cache'
    ARTIFACT_FORMAT = 'sqlite3'

    def __init__(
        self,
        path: str,
        transformer: pt.Transformer = None,
        *,
        key: Optional[Union[str, List[str], Tuple[str]]] = None,
        value: Optional[Union[str, List[str], Tuple[str]]] = None,
        verbose: bool = False,
    ):
        """
        Args:
            path: The path to the directory where the cache should be stored.
            transformer: The transformer to apply to get misses from the cache when value isn't found for key.
            key: The name of the column(s) in the input DataFrame that contains the key for looking up value. Can be omitted for caches that have already been created.
            value: The name of the column(s) in the input DataFrame that contains the value value to cache. Can be omitted for caches that have already been created.
            verbose: Whether to print verbose output when scoring documents.

        If a cache does not yet exist at the provided ``path``, a new one is created.

        .. versionadded:: 0.4.0
        """
        super().__init__(path)
        self.transformer = transformer
        self.verbose = verbose
        self.meta = None

        if key is not None:
            if isinstance(key, str):
                key = (key,)
            key = list(key)
        if value is not None:
            if isinstance(value, str):
                value = (value,)
            value = list(value)

        if not (Path(self.path)/'pt_meta.json').exists():
            assert key is not None, "key must be provided when creating a new cache"
            assert value is not None, "value must be provided when creating a new cache"
            with pta.ArtifactBuilder(self) as builder:
                builder.metadata['key'] = key
                builder.metadata['value'] = value
                self.db = sqlite3.connect(builder.path/'db.sqlite3')
                with closing(self.db.cursor()) as cursor:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS cache (
                          key BLOB NOT NULL,
                          value BLOB NOT NULL,
                          PRIMARY KEY (key)
                        )
                    """)
        else:
            self.db = sqlite3.connect(self.path/'db.sqlite3')
        with (Path(self.path)/'pt_meta.json').open('rt') as fin:
            self.meta = json.load(fin)

        if key is None:
            key = self.meta['key']
        assert key == self.meta['key'], f'key={key!r} provided, but index created with key={self.meta["key"]!r}'
        self.key = key

        if value is None:
            value = self.meta['value']
        assert value == self.meta['value'], f'value={value!r} provided, but index created with value={self.meta["value"]!r}'
        self.value = value

    def close(self):
        """ Closes this cache, releasing the sqlite connection that it holds. """
        if self.db is not None:
            self.db.close()
            self.db = None

    @contextmanager
    def _cursor(self):
        assert self.db is not None, "cache is closed"
        with closing(self.db.cursor()) as cursor:
            yield cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.columns(inp, includes=list(self.key))

        to_score_idxs = []
        to_score_map = {}

        inp = inp.reset_index(drop=True)
        keys = [pickle.dumps(tuple(tup), protocol=PICKLE_PROTOCOL) for tup in inp[self.key].itertuples(index=False)]
        values = [[None] * len(inp) for _ in self.value]

        # First pass: load what we can from cache
        key2idxs = defaultdict(list)
        for idx, key in enumerate(keys):
            key2idxs[key].append(idx)
        with self._cursor() as cursor:
            placeholder = ', '.join(['?'] * len(keys))
            cursor.execute(f'SELECT key, value FROM cache WHERE key IN ({placeholder})', keys)
            for key, value in cursor.fetchall():
                value = pickle.loads(value)
                for idx in key2idxs[key]:
                    for v, vs in zip(value, values):
                        vs[idx] = v
                del key2idxs[key]
        for key, idxs in key2idxs.items():
            to_score_idxs.extend(idxs)
            to_score_map[key] = idxs

        # Second pass: score the missing ones and add to cache
        if to_score_idxs:
            if self.transformer is None:
                raise LookupError('values missing from cache, but no transformer provided')
            scored = self.transformer(inp.loc[to_score_idxs])
            keys = scored[self.key]
            keys_enc = [pickle.dumps(tuple(k), protocol=PICKLE_PROTOCOL) for k in keys.itertuples(index=False)]
            value = scored[self.value]
            value_enc = [pickle.dumps(tuple(v), protocol=PICKLE_PROTOCOL) for v in value.itertuples(index=False)]
            with closing(self.db.cursor()) as cursor:
                cursor.executemany('INSERT INTO cache (key, value) VALUES (?, ?)', list(zip(keys_enc, value_enc)))
                self.db.commit()
            for key_enc, value in zip(keys_enc, value.itertuples(index=False)):
                for idx in to_score_map[key_enc]:
                    for v, vs in zip(value, values):
                        vs[idx] = v

        results = inp.assign(**{n: v for n, v in zip(self.value, values)})
        if self.verbose:
            print(f"{self}: {len(inp)-len(to_score_idxs)} hit(s), {len(to_score_idxs)} miss(es)")
        return results

    def __iter__(self):
        with closing(self.db.cursor()) as cursor:
            for key, value in cursor.execute('SELECT key, value FROM cache').fetchall():
                key = pickle.loads(key)
                value = pickle.loads(value)
                res = {}
                res.update(zip(self.key, key))
                res.update(zip(self.value, value))
                yield res

    def __repr__(self):
        return f'Sqlite3KeyValueCache({str(self.path)!r}, {self.transformer!r}, key={self.key!r}, value={self.value!r})'

# Default implementations
KeyValueCache = Sqlite3KeyValueCache
