from typing import Optional, List, Iterator, Dict, Any, Union, Literal
from pathlib import Path
from contextlib import ExitStack
import pickle
import json
import lz4.frame
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import pyterrier as pt
from npids import Lookup
from tqdm import tqdm
import pyterrier_alpha as pta
from pyterrier_caching import closing_memmap, meta_file_compat


class Lz4PickleIndexerCache(pta.Artifact, pt.Indexer):
    """An :class:`~pyterrier_caching.IndexerCache` that stores records as pickled dictionaries compressed with lz4.
    """
    ARTIFACT_TYPE = 'indexer_cache'
    ARTIFACT_FORMAT = 'lz4pickle'

    def __init__(self, path: Optional[str] = None):
        """
        Args:
            path: The path to the cache. If None, a temporary cache will be created that is deleted when closed.
        """
        if path is None:
            self._tmpdir = TemporaryDirectory()
            path = Path(self._tmpdir.name) / 'cache'
        else:
            self._tmpdir = None
        super().__init__(path)
        meta_file_compat(path)
        self._docnos = None

    def indexer(self, mode: Union[str, pta.ArtifactBuilderMode] = pta.ArtifactBuilderMode.create, skip_docno_lookup: bool = False) -> pt.Indexer:
        """Returns an :class:`~pyterrier.Indexer` for this cache. The indexer can be used to create the cache.

        Args:
            mode: The mode to use for the indexer. Must be 'create'.
            skip_docno_lookup: If True, skip creating a docno lookup.
        """
        return Lz4PickleIndexerCacheIndexer(self, mode, skip_docno_lookup=skip_docno_lookup)

    def index(self, it: Iterator[Dict[str, Any]]) -> None:
        """Indexes the provided records to this cache."""
        return self.indexer().index(it)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterates over the records stored in the cache."""
        return self.get_corpus_iter()

    def __len__(self) -> int:
        """Returns the number of records stored in the cache."""
        if not self.built():
            raise RuntimeError('cache not built')
        with (Path(self.path)/'pt_meta.json').open('rt') as fin:
            metadata = json.load(fin)
        return metadata['record_count']

    def get_corpus_iter(self, verbose: bool = False, fields: Optional[List[str]] = None, start: Optional[int] = None, stop: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Iterates over the records stored in the cache.

        Args:
            verbose: If True, show a progress bar.
            fields: If not None, only return these fields.
            start: If not None, start at this record number.
            stop: If not None, stop at this record number.
        """
        # validate arguments
        if fields is not None:
            fields = set(fields)
        if start is not None:
            assert start >= 0
        if stop is not None:
            assert stop >= 0
            # we take one extra one to get the offset of the last record, too.
            stop += 1

        with open(Path(self.path)/'data.pkl.lz4', mode='rb') as fdata, \
             closing_memmap(Path(self.path)/'offsets.np', dtype=np.uint64) as offsets_mmp:
            offsets = offsets_mmp[start:stop]
            if len(offsets) > 0:
                fdata.seek(offsets[0])
                it = zip(offsets[:-1], offsets[1:])
                if verbose:
                    it = tqdm(it, total=len(offsets)-1, unit='d')
                for offset_start, offset_stop in it:
                    record_length = offset_stop - offset_start
                    record = fdata.read(record_length)
                    record = lz4.frame.decompress(record)
                    record = pickle.loads(record)
                    if fields is not None:
                        record = {k: v for k, v in record.items() if k in fields}
                    yield record

    def to_dataframe(self, verbose: bool = False, fields: Optional[List[str]] = None, start: Optional[int] = None, stop: Optional[int] = None) -> pd.DataFrame:
        """Converts the results in this cache to a DataFrame.

        Args:
            verbose: If True, show a progress bar.
            fields: If not None, only return these fields.
            start: If not None, start at this record number.
            stop: If not None, stop at this record number.
        """
        return pd.DataFrame(self.get_corpus_iter(verbose=verbose, fields=fields, start=start, stop=stop))

    def __getitem__(self, items: Union[int, str, slice]):
        """Returns the record(s) stored in the cache by the provided index, docno, or range.
        """
        if isinstance(items, int):
            return next(self.get_corpus_iter(start=items, stop=items+1))
        elif isinstance(items, str):
            idx = Lookup(Path(self.path)/'docnos.npids').inv[items]
            return self[idx]
            # TODO: support lookup by a list of strings?
        elif isinstance(items, slice):
            assert items.step is None, "step is not supported by Lz4PickleIndexerCache"
            return list(self.get_corpus_iter(start=items.start, stop=items.stop))
        raise ValueError('unknown type for items: {}'.format(type(items)))

    def built(self) -> bool:
        """Returns True if the cache is built."""
        return (Path(self.path)/'pt_meta.json').exists()

    def text_loader(
        self,
        fields: Union[str, List[str], Literal['*']] = '*',
        *,
        verbose: bool = False,
    ) -> pt.Transformer:
        """Returns a :class:`~pyterrier.Transformer` that loads the text from the cache based on ``docno``.

        Args:
            fields: If not '*', only return these fields.
            verbose: If True, show a progress bar.
        """
        return Lz4PickleIndexerCacheTextLoader(self, fields, verbose=verbose)

    def docnos(self):
        """Returns a :class:`~npids.Lookup` for the docnos stored in the cache."""
        assert self.built()
        if self._docnos is None:
            self._docnos = Lookup(Path(self.path)/'docnos.npids')
        return self._docnos

    def close(self):
        """Closes any open files used by this cache."""
        if self._docnos is not None:
            self._docnos.close()
            self._docnos = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class Lz4PickleIndexerCacheIndexer(pt.Indexer):
    def __init__(self, cache: Lz4PickleIndexerCache, mode: Union[str, pta.ArtifactBuilderMode], skip_docno_lookup: bool = False):
        self.cache = cache
        self.mode = mode
        self.skip_docno_lookup = skip_docno_lookup

    def index(self, it: Iterator[Dict[str, Any]]) -> None:
        assert pta.ArtifactBuilderMode(self.mode) == pta.ArtifactBuilderMode.create, "Lz4PickleIndexerCache only supports 'create' mode"
        with ExitStack() as s:
            builder = s.enter_context(pta.ArtifactBuilder(self.cache, mode=self.mode))
            fdata = s.enter_context(open(builder.path/'data.pkl.lz4', mode='wb'))
            foffsets = s.enter_context(open(builder.path/'offsets.np', 'wb'))
            docno_lookup = False if self.skip_docno_lookup else None
            foffsets.write(np.array(0, dtype=np.uint64).tobytes())
            builder.metadata['record_count'] = 0
            for record in it:
                if docno_lookup is None:
                    if 'docno' in record:
                        docno_lookup = s.enter_context(Lookup.builder(builder.path/'docnos.npids'))
                if docno_lookup is not False:
                    docno_lookup.add(record['docno'])
                record_bytes = pickle.dumps(dict(record))
                fdata.write(lz4.frame.compress(record_bytes))
                foffsets.write(np.array(fdata.tell(), dtype=np.uint64))
                builder.metadata['record_count'] += 1


class Lz4PickleIndexerCacheTextLoader(pt.Transformer):
    def __init__(self,
        cache: Lz4PickleIndexerCache,
        fields: Union[str, List[str], Literal['*']] = '*',
        *,
        verbose: bool = False,
    ):
        self.cache = cache
        self.fields = fields
        self.verbose = verbose

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.columns(inp, includes=['docno'])
        if len(inp) == 0:
            return inp

        if self.fields == '*':
            fields = None
        elif isinstance(self.fields, str):
            fields = [self.fields]
        else:
            fields = self.fields

        doc_ids = self.cache.docnos().inv[inp['docno'].to_list()]

        builder = None
        with open(Path(self.cache.path)/'data.pkl.lz4', mode='rb') as fdata, \
             closing_memmap(Path(self.cache.path)/'offsets.np', dtype=np.uint64) as offsets_mmp:
            if self.verbose:
                doc_ids = tqdm(doc_ids, total=len(doc_ids), unit='d')
            for did in doc_ids:
                offset_start, offset_stop = offsets_mmp[did:did+2]
                record_length = offset_stop - offset_start
                fdata.seek(offset_start)
                record = fdata.read(record_length)
                record = lz4.frame.decompress(record)
                record = pickle.loads(record)
                if fields is not None:
                    record = {k: v for k, v in record.items() if k in fields}
                record.pop('docno')
                if builder is None:
                    builder = pta.DataFrameBuilder(list(record.keys()))
                builder.extend(record)

        return builder.to_df(inp)


# Default implementation of IndexerCache: Lz4PickleIndexerCache
IndexerCache = Lz4PickleIndexerCache
