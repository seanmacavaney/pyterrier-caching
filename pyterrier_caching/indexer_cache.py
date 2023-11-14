from typing import Optional, List, Iterator, Dict, Any, Union
from pathlib import Path
from contextlib import ExitStack
import struct
import pickle
import lz4.frame
import numpy as np
import pandas as pd
import pyterrier as pt
from npids import Lookup
from pyterrier_caching import BuilderMode, closing_memmap, artefact_builder


class Lz4PickleIndexerCache(pt.Indexer):
    artefact_type = 'indexer_cache'
    artefact_format = 'lz4pickle'

    def __init__(self, path: str):
        self.path = path

    def indexer(self, mode: Union[str, BuilderMode] = BuilderMode.create, skip_docno_lookup: bool = False) -> pt.Indexer:
        return Lz4PickleIndexerCacheIndexer(self, mode, skip_docno_lookup=skip_docno_lookup)

    def index(self, it: Iterator[Dict[str, Any]]) -> None:
        return self.indexer().index(it)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self.get_corpus_iter()

    def get_corpus_iter(self, verbose: bool = False, fields: Optional[List[str]] = None, start: Optional[int] = None, stop: Optional[int] = None) -> Iterator[Dict[str, Any]]:
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
        return pd.DataFrame(self.get_corpus_iter(verbose=verbose, fields=fields, start=start, stop=stop))

    def __getitem__(self, items: Union[int, str, slice]):
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
        return (Path(self.path)/'meta.json').exists()


class Lz4PickleIndexerCacheIndexer(pt.Indexer):
    def __init__(self, cache: Lz4PickleIndexerCache, mode: Union[str, BuilderMode], skip_docno_lookup: bool = False):
        self.cache = cache
        self.mode = mode
        self.skip_docno_lookup = skip_docno_lookup

    def index(self, it: Iterator[Dict[str, Any]]) -> None:
        assert BuilderMode(self.mode) == BuilderMode.create, "Lz4PickleIndexerCache only supports 'create' mode"
        with ExitStack() as s:
            builder = s.enter_context(artefact_builder(self.cache.path, self.mode, self.cache.artefact_type, self.cache.artefact_format))
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


# Default implementation of IndexerCache: Lz4PickleIndexerCache
IndexerCache = Lz4PickleIndexerCache
