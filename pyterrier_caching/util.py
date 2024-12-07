from pathlib import Path
from contextlib import contextmanager
import numpy as np
import pandas as pd
import pyterrier as pt

@contextmanager
def closing_memmap(*args, **kwargs):
    """A context manager that creates a :class:`numpy.memmap` and closes it when the context is exited.

    This allows :class:`numpy.memmap` to be used as a context manager, since it doesn't support the
    context manager protocol directly.

    Args:
        *args: Positional arguments to pass to :class:`numpy.memmap`.
        **kwargs: Keyword arguments to pass to :class:`numpy.memmap`.
    
    Example:

    .. code-block:: python
        :caption: Using a :func:`~pyterrier_caching.util.closing_memmap` context manager.

        from pyterrier_caching import closing_memmap
        with closing_memmap('file.npy', dtype='float32', mode='w+', shape=(10, 10)) as mmp:
            # do what you want with mmp here!
        # mmp is closed here
    """
    mmp = None
    try:
        mmp = np.memmap(*args, **kwargs)
        yield mmp
    finally:
        if mmp is not None:
            del mmp
            mmp = None


class Lazy(pt.Transformer):
    """A :class:`~pyterrier.Transformer` that doesn't initialize until it is used.

    This is useful in cases where loading a transformer is lengthy or allocates resources that are not always
    necessary. For instance a cached neural neural scorer allocates GPU memory, but often isn't needed when used
    with a :class:`~pyterrier_caching.ScorerCache`.

    Example:

    .. code-block:: python
        :caption: Using a :class:`~pyterrier_caching.Lazy` :class:`~pyterrier_dr.ElectraScorer` with a :class:`~pyterrier_caching.ScorerCache`.

        from pyterrier_caching import ScorerCache
        from pyterrier_dr import ElectraScorer
        lazy_scorer = Lazy(ElectraScorer) # ElectraScorer not loaded yet
        cached_scorer = ScorerCache('electra.cache', lazy_scorer)
        cached_scorer([{
            'qid': '0',
            'query': 'terrier breeds',
            'docno': 'doc1',
            'text': 'There are many breeds of terriers, including the Scottish and Jack Russell Terrier.'
        ])
        # ElectraScorer only loaded if ('0', 'doc1') is not yet in electra.cache
    """
    def __init__(self, fn_transformer: pt.Transformer, *fn_args, **fn_kwargs):
        """
        Args:
            fn_transformer: A function that returns a transformer when called (or the transformer class itself).
            fn_args: Positional arguments to pass to ``fn_transformer`` when loading it.
            fn_kwargs: Keyword arguments to pass to ``fn_transformer`` when loading it.
        """
        self.fn_transformer = fn_transformer
        self.fn_args = fn_args
        self.fn_kwargs = fn_kwargs
        self._transformer = None

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        return self.load()(inp)

    def load(self) -> pt.Transformer:
        """Load the transformer if it isn't already loaded, and return it."""
        if not self.loaded():
            self._transformer = self.fn_transformer(*self.fn_args, **self.fn_kwargs)
        return self._transformer

    def unload(self):
        """Unloads the transformer. Subsequent calls to :meth:`load` will re-load it."""
        self._transformer = None

    def loaded(self) -> bool:
        """Return whether the transformer is currently loaded."""
        return self._transformer is not None


def meta_file_compat(path):
    """
    Until version 0.1.0, pt_meta.json was called meta.json. To ensure compatiblity between caches created with
    version <0.1.0 and >=0.1.0, this method moves meta.json to pt_meta.json and linkns meta.json -> pt_meta.json.

    The end effect is that caches created with version <0.1.0 will be compatible with >=0.1.0, but caches created
    with >=0.1.0 will NOT be compatible with those created with <0.1.0.
    """
    path = Path(path)
    if (old_path := (path/'meta.json')).exists() and \
       not (new_path := (path/'pt_meta.json')).exists():
        old_path.rename(new_path)
        old_path.symlink_to(new_path)
