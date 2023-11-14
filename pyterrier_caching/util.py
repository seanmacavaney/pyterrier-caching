from typing import Union
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
