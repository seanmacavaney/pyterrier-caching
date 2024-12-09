from typing import Optional
from pathlib import Path
import hashlib
import pickle
import numpy as np
import pandas as pd
import pyterrier as pt
import shutil
import json
import sqlite3
from collections import defaultdict
from contextlib import closing, contextmanager
from more_itertools import chunked
from npids import Lookup
from pyterrier_caching import meta_file_compat
import pyterrier_alpha as pta


class Hdf5ScorerCache(pta.Artifact, pt.Transformer):
    """A cache for storing and retrieving scores for documents, backed by an HDF5 file. 

    This is a *dense* scorer cache, meaning that space for all documents is allocated ahead of time.
    Dense caches are more suitable than sparse ones when a large proportion of the corpus (or
    the entire corpus) is expected to be scored. If only a small proportion of the corpus is expected
    to be scored, a sparse cache (e.g., :class:`~pyterrier_caching.Sqlite3ScorerCache`) may be more appropriate.
    """
    ARTIFACT_TYPE = 'scorer_cache'
    ARTIFACT_FORMAT = 'hdf5'

    def __init__(self, path, scorer=None, verbose=False):
        """
        Args:
            path: The path to the directory where the cache should be stored.
            scorer: The scorer to use to score documents that are missing from the cache.
            verbose: Whether to print verbose output when scoring documents.
        """
        super().__init__(path)
        meta_file_compat(path)
        self.scorer = scorer
        self.verbose = verbose
        self.meta = None
        self.file = None
        self.docnos = None
        self.dataset_cache = {}

    def transform(self, inp):
        return self.cached_scorer()(inp)

    def built(self) -> bool:
        """Returns whether this cache has been built."""
        return (Path(self.path)/'pt_meta.json').exists()

    def build(self, corpus_iter=None, docnos_file=None):
        """Builds this cache."""
        assert not self.built(), "this cache is alrady built"
        assert corpus_iter is not None or docnos_file is not None
        import h5py
        with pta.ArtifactBuilder(self) as builder:
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
            self.file = h5py.File(self.path/'data.h5', 'r')
        if self.meta is None:
            with (self.path/'pt_meta.json').open('rt') as fin:
                self.meta = json.load(fin)
        if self.docnos is None:
            self.docnos = Lookup(self.path/'docnos.npids')

    def _get_dataset(self, qid):
        if qid not in self.dataset_cache:
            if qid not in self.file:
                return None
            self.dataset_cache[qid] = self.file[qid][:]
        return self.dataset_cache[qid]

    def corpus_count(self) -> int:
        """Returns the number of documents in the corpus that this cache was built from."""
        self._ensure_built()
        return self.meta['doc_count']

    def __repr__(self):
        return f'Hdf5ScorerCache({repr(str(self.path))}, {self.scorer})'

    def cached_scorer(self) -> pt.Transformer:
        """Returns a scorer that uses this cache to store and retrieve scores."""
        return Hdf5ScorerCacheScorer(self)

    def cached_retriever(self, num_results: int = 1000) -> pt.Transformer:
        """Returns a retriever that uses this cache to store and retrieve scores for every document in the corpus.

        This transformer will raie an error if the entire corpus is not scored (e.g., from :meth:`score_all`).
        """
        return Hdf5ScorerCacheRetriever(self, num_results)

    def close(self):
        """ Closes this cache, releasing the file pointer that it holds and writing any new results to disk. """
        if self.file is not None:
            self.dataset_cache = {} # reset the dataset cache
            self.file.close()
            self.file = None
        if self.meta is not None:
            self.meta = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def merge_from(self, other: 'Hdf5ScorerCache'):
        """ Merges the cached values from another Hdf5ScorerCache instance into this one.

        Any records that appear in both ``self`` and ``other`` will be replaced with the value from ``other``.
        """
        count = 0
        other._ensure_built()
        self._ensure_built()
        self._ensure_write_mode()
        assert self.meta['doc_count'] == other.meta['doc_count'], f'{self} and {other} are incompatible'
        for key in other.file.keys():
            data = other[key][:]
            if key in self.file:
                # copy over the non-missing values
                mask = ~np.isnan(data)
                self.file[key][mask] = data[mask]
                if self.verbose:
                    count += mask.sum()
            else:
                # easy: just copy it all over
                self.file.create_dataset(key, data=data)
                if self.verbose:
                    count += (~np.isnan(data)).sum()
        if self.verbose:
            print(f"merged {count} records from {other} into {self}")

    def score_all(self, dataset, *, batch_size: int = 1024):
        """Scores all topics for the entire corpus, storing the results in this cache."""
        if not self.built():
            self.build(dataset.get_corpus_iter())
        topics = dataset.get_topics()
        scorer = self.cached_scorer()
        for doc_batch in chunked(dataset.get_corpus_iter(), batch_size):
            doc_batch = pd.DataFrame(doc_batch)
            inp = pd.merge(topics, doc_batch, how='cross')
            scorer(inp)


class Hdf5ScorerCacheScorer(pt.Transformer):
    def __init__(self, cache: Hdf5ScorerCache):
        self.cache = cache

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        import h5py
        self.cache._ensure_built()

        to_score_idxs = []
        to_score_map = defaultdict(list)

        inp = inp.reset_index(drop=True)
        values = pd.Series(index=inp.index, dtype=float)

        # First pass: load what we can from cache
        for query, group in inp.groupby('query'):
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            ds = self.cache._get_dataset(query_hash)
            if ds is None:
                for idx in group.index:
                    to_score_idxs.append(idx)
                    to_score_map[query_hash, group.docno[idx]].append(idx)
            else:
                dids = self.cache.docnos.inv[np.array(group.docno)]
                assert not (dids == -1).any(), 'unknown docno encountered'
                scores = ds[dids]
                for idx, score, is_miss in zip(group.index, scores, np.isnan(scores)):
                    if is_miss:
                        to_score_idxs.append(idx)
                        to_score_map[query_hash, group.docno[idx]].append(idx)
                    else:
                        values[idx] = score

        # Second pass: score the missing ones and add to cache
        if to_score_idxs:
            if self.cache.scorer is None:
                raise LookupError('values missing from cache, but no scorer provided')
            scored = self.cache.scorer(inp.loc[to_score_idxs])
            self.cache.close()
            with h5py.File(self.cache.path/'data.h5', 'a') as fout:
                records = scored[['query', 'docno', 'score']]
                for query, group in records.groupby('query'):
                    query_hash = hashlib.sha256(query.encode()).hexdigest()
                    if query_hash not in fout:
                        fout.create_dataset(query_hash, shape=(self.cache.corpus_count(),), dtype=np.float32, fillvalue=float('nan'))
                    dids = self.cache.docnos.inv[np.array(group.docno)]
                    assert not (dids == -1).any(), 'unknown docno encountered'
                    dids_sorted, dids_sorted_idx = np.unique(dids, return_index=True)
                    fout[query_hash][dids_sorted] = group.score.iloc[dids_sorted_idx]
                    for _, docno, score in group.itertuples(index=False):
                        for idx in to_score_map[query_hash, docno]:
                            values[idx] = score

        results = inp.assign(score=values)
        pt.model.add_ranks(results)
        if self.cache.verbose:
            print(f"{self}: {len(inp)-len(to_score_idxs)} hit(s), {len(to_score_idxs)} miss(es)")
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
            ds = self.cache._get_dataset(query_hash)
            if ds is None:
                raise RuntimeError(f'retriever only works if corpus is scored completely; '
                                   f'no cached documents found for query {query!r}.')
            ds = ds[:]
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
                'score': scores[idxs].astype(np.float64),
                'rank': np.arange(scores.shape[0]),
            })
        return builder.to_df(merge_on_index=inp)


class Sqlite3ScorerCache(pta.Artifact, pt.Transformer):
    """A cache for storing and retrieving scores for documents, backed by a SQLite3 database.

    This is a *sparse* scorer cache, meaning that space is only allocated for documents that have been scored.
    If a large proportion of the corpus is expected to be scored, a dense cache (e.g., :class:`~pyterrier_caching.Hdf5ScorerCache`)
    may be more appropriate.
    """

    ARTIFACT_TYPE = 'scorer_cache'
    ARTIFACT_FORMAT = 'sqlite3'

    def __init__(
        self,
        path: str,
        scorer: pt.Transformer = None,
        *,
        group: Optional[str] = None,
        key: Optional[str] = None,
        value: Optional[str] = None,
        pickle : Optional[bool] = None,
        verbose: bool = False,
    ):
        """
        Args:
            path: The path to the directory where the cache should be stored.
            scorer: The scorer to use to score documents that are missing from the cache.
            group: The name of the column in the input DataFrame that contains the group identifier (default: ``query``)
            key: The name of the column in the input DataFrame that contains the document identifier (default: ``docno``)
            value: The name of the column in the input DataFrame that contains the value to cache (default: ``score``)
            pickle: Whether to pickle the value before storing it in the cache (default: False)
            verbose: Whether to print verbose output when scoring documents.

        If a cache does not yet exist at the provided ``path``, a new one is created.

        .. versionchanged:: 0.3.0 added ``pickle`` option to support caching non-numeric values
        """
        super().__init__(path)
        meta_file_compat(path)
        self.scorer = scorer
        self.verbose = verbose
        self.meta = None
        if not (Path(self.path)/'pt_meta.json').exists():
            if group is None:
                group = 'query'
            if key is None:
                key = 'docno'
            if value is None:
                value = 'score'
            with pta.ArtifactBuilder(self) as builder:
                builder.metadata['group'] = group
                builder.metadata['key'] = key
                builder.metadata['value'] = value
                builder.metadata['pickle'] = pickle
                self.db = sqlite3.connect(builder.path/'db.sqlite3')
                value_type = "BLOB" if pickle else "NUMERIC"
                with closing(self.db.cursor()) as cursor:
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS cache (
                          [group] TEXT NOT NULL,
                          key TEXT NOT NULL,
                          value {value_type} NOT NULL,
                          PRIMARY KEY ([group], key)
                        )
                    """)
        else:
            self.db = sqlite3.connect(self.path/'db.sqlite3')
        with (Path(self.path)/'pt_meta.json').open('rt') as fin:
            self.meta = json.load(fin)
        self.meta.setdefault('pickle', False)
        if group is not None:
            assert group == self.meta['group'], f'group={group!r} provided, but index created with group={self.meta["group"]!r}'
        self.group = self.meta['group']
        if key is not None:
            assert key == self.meta['key'], f'key={key!r} provided, but index created with key={self.meta["key"]!r}'
        self.key = self.meta['key']
        if value is not None:
            assert value == self.meta['value'], f'value={value!r} provided, but index created with value={self.meta["value"]!r}'
        self.value = self.meta['value']
        if pickle is not None:
            assert pickle == self.meta['pickle'], f'pickle={pickle!r} provided, but index created with pickle={self.meta["pickle"]!r}'
        self.pickle = self.meta['pickle']

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
        """ Scores the input DataFrame using cached values, scoring any missing ones and adding them to the cache. """
        pta.validate.columns(inp, includes=[self.group, self.key])

        to_score_idxs = []
        to_score_map = {}

        inp = inp.reset_index(drop=True)
        values = pd.Series(index=inp.index, dtype=object if self.pickle else float)

        # First pass: load what we can from cache
        for group_key, group in inp.groupby(self.group):
            placeholder = ', '.join(['?'] * len(group))
            key2idxs = defaultdict(list)
            for idx, key in zip(group.index, group[self.key]):
                key2idxs[key].append(idx)
            with self._cursor() as cursor:
                cursor.execute(f'SELECT key, value FROM cache WHERE [group]=? AND key IN ({placeholder})',
                    [group_key] + group[self.key].tolist())
                for key, score in cursor.fetchall():
                    for idx in key2idxs[key]:
                        values[idx] = pickle.loads(score) if self.pickle else score 
                    del key2idxs[key]
            for key, idxs in key2idxs.items():
                to_score_idxs.extend(idxs)
                to_score_map[group_key, key] = idxs

        # Second pass: score the missing ones and add to cache
        if to_score_idxs:
            if self.scorer is None:
                raise LookupError('values missing from cache, but no scorer provided')
            scored = self.scorer(inp.loc[to_score_idxs])
            records = scored[[self.group, self.key, self.value]]
            rec_it = records.itertuples(index=False)
            if self.pickle:
                rec_it = [(g, k, pickle.dumps(v)) for g, k, v in rec_it]
            with closing(self.db.cursor()) as cursor:
                cursor.executemany('INSERT INTO cache ([group], key, value) VALUES (?, ?, ?)', rec_it)
                self.db.commit()
            for group, key, score in records.itertuples(index=False):
                for idx in to_score_map[group, key]:
                    values[idx] = score

        results = inp.assign(**{self.value: values})
        if self.value == 'score':
            pt.model.add_ranks(results)
        if self.verbose:
            print(f"{self}: {len(inp)-len(to_score_idxs)} hit(s), {len(to_score_idxs)} miss(es)")
        return results

    def merge_from(self, other: 'Sqlite3ScorerCache'):
        """ Merges the cached values from another Sqlite3ScorerCache instance into this one.

        Any keys that appear in both ``self`` and ``other`` will be replaced with the value from ``other``.
        """
        count = 0
        with self._cursor() as insert_cursor, other._cursor() as select_cursor:
            select_cursor.execute('SELECT [group], key, value FROM cache')
            while batch := select_cursor.fetchmany(10_000):
                count += len(batch)
                insert_cursor.executemany('INSERT OR REPLACE INTO cache ([group], key, value) VALUES (?, ?, ?)', batch)
            self.db.commit()
        if self.verbose:
            print(f"merged {count} records from {other} into {self}")

    def __repr__(self):
        return f'Sqlite3ScorerCache({str(self.path)!r}, {self.scorer!r}, group={self.group!r}, key={self.key!r})'

# Default implementations
ScorerCache = Sqlite3ScorerCache
DenseScorerCache = Hdf5ScorerCache
SparseScorerCache = Sqlite3ScorerCache
