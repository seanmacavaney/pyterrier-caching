import tempfile
import unittest
from pathlib import Path
import pyterrier as pt
import pyterrier_caching

class TestIndexerCache(unittest.TestCase):
    def setUp(self):
        if not pt.started():
            pt.init(version='5.8', helper_version='0.0.8')

    def test_basic(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cache = pyterrier_caching.IndexerCache(d/'cache')
            cache.index([
                {'docno': '1', 'data': 'test'},
                {'docno': '2', 'data': 'caching pyterrier'},
                {'docno': '3', 'data': 'fetch me some data'},
                {'docno': '4', 'data': 'information retrieval'},
                {'docno': '5', 'data': 'foo bar baz'},
            ])
            self.assertEqual(list(cache), [
                {'docno': '1', 'data': 'test'},
                {'docno': '2', 'data': 'caching pyterrier'},
                {'docno': '3', 'data': 'fetch me some data'},
                {'docno': '4', 'data': 'information retrieval'},
                {'docno': '5', 'data': 'foo bar baz'},
            ])
            self.assertEqual(list(cache.get_corpus_iter(start=2)), [
                {'docno': '3', 'data': 'fetch me some data'},
                {'docno': '4', 'data': 'information retrieval'},
                {'docno': '5', 'data': 'foo bar baz'},
            ])
            self.assertEqual(list(cache.get_corpus_iter(stop=2)), [
                {'docno': '1', 'data': 'test'},
                {'docno': '2', 'data': 'caching pyterrier'},
            ])
            self.assertEqual(list(cache.get_corpus_iter(start=1, stop=3)), [
                {'docno': '2', 'data': 'caching pyterrier'},
                {'docno': '3', 'data': 'fetch me some data'},
            ])

    def test_empty(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cache = pyterrier_caching.IndexerCache(d/'cache')
            cache.index(range(0))
            self.assertEqual(list(cache), [])
