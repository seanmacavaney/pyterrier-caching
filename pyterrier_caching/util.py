from pathlib import Path
from contextlib import contextmanager
import numpy as np
import pyterrier as pt

@contextmanager
def closing_memmap(*args, **kwargs):
    # np.memmap isn't allowed to be used as a context manager directly,
    # and the proper way to close it is by deleting the object.
    # This is a context manager that does that.
    mmp = None
    try:
        mmp = np.memmap(*args, **kwargs)
        yield mmp
    finally:
        if mmp is not None:
            del mmp
            mmp = None


class Lazy(pt.Transformer):
    def __init__(self, fn_transformer):
        self.fn_transformer = fn_transformer
        self.transformer = None

    def transform(self, inp):
        if not self.loaded():
            self.transformer = self.fn_transformer()
        return self.transformer(inp)

    def loaded(self):
        return self.transformer is not None

    def unload(self):
        self.transformer = None


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
