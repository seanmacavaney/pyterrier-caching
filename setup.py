import setuptools


def get_version(path):
    for line in open(path, 'rt'):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name='pyterrier-caching',
    version=get_version('pyterrier_caching/__init__.py'),
    author='Sean MacAvaney',
    author_email='sean.macavaney@glasgow.ac.uk',
    description='Caching components for PyTerrier',
    long_description=open('README.md', 'rt').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=list(open('requirements.txt', 'rt')),
    python_requires='>=3.8',
    entry_points={
        'pyterrier.artifact': [
            'indexer_cache.lz4pickle = pyterrier_caching:Lz4PickleIndexerCache',
            'retriever_cache.dbm.dumb = pyterrier_caching:DbmRetrieverCache',
            'scorer_cache.hdf5 = pyterrier_caching:Hdf5ScorerCache',
        ],
    },
)
