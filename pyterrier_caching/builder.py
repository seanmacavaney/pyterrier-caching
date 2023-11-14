from typing import Union
import json
from enum import Enum
from pathlib import Path
from contextlib import contextmanager

class BuilderMode(Enum):
    create = 'create'
    overwrite = 'overwrite'
    append = 'append'


class ArtefactBuilderState:
    def __init__(self, path, metadata=None):
        self.path = Path(path)
        self.metadata = metadata or {}


@contextmanager
def artefact_builder(path: Union[str, Path], mode: Union[BuilderMode, str], artefact_type: str, artefact_format: str):
    mode = BuilderMode(mode)
    path = Path(path)

    state = ArtefactBuilderState(path, metadata={
        'type': artefact_type,
        'format': artefact_format,
    })

    # TODO: check if the path exists and either:
    if mode == BuilderMode.create:
        if path.exists():
            raise FileExistsError(f'{str(path)} already exists.')
        path.mkdir(parents=True)
    elif mode == BuilderMode.overwrite:
        if path.exists():
            raise NotImplementedError()
            # This might be a bit tricky... Do we recursively delete the entire
            # directory if it exists? What if the user accidently enters a path
            # such as '/'? Do we need some sort of check or warning if the path
            # is very large? Hmmm...
        else:
            path.mkdir(parents=True)
    elif mode == BuilderMode.append:
        raise NotImplementedError()

    try:
        yield state
    except:
        # TODO: clean up
        raise

    # Log the artefact metadata
    meta_path = path / 'meta.json'
    try:
        with open(meta_path, 'wt') as fout:
            json.dump(state.metadata, fout)
    except:
        if meta_path.exists():
            meta_path.unlink()
        raise
