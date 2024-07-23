from typing import Union
import json
from enum import Enum
from pathlib import Path
from contextlib import contextmanager


class BuilderMode(Enum):
    create = 'create'
    overwrite = 'overwrite'
    append = 'append'


class ArtifactBuilderState:
    def __init__(self, path, metadata=None):
        self.path = Path(path)
        self.metadata = metadata or {}


@contextmanager
def artifact_builder(path: Union[str, Path], mode: Union[BuilderMode, str], artifact_type: str, artifact_format: str):
    mode = BuilderMode(mode)
    path = Path(path)

    state = ArtifactBuilderState(path, metadata={
        'type': artifact_type,
        'format': artifact_format,
        'package_hint': 'pyterrier-caching',
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

    # Log the artifact metadata
    meta_path = path / 'pt_meta.json'
    try:
        with open(meta_path, 'wt') as fout:
            json.dump(state.metadata, fout)
    except:
        if meta_path.exists():
            meta_path.unlink()
        raise
