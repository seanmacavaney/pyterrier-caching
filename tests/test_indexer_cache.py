import tempfile
import unittest
from pathlib import Path
import pandas as pd
import pyterrier_caching

class TestIndexerCache(unittest.TestCase):
    def test_basic(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cache = pyterrier_caching.IndexerCache(d/'cache')
            with self.subTest('len before built'):
                with self.assertRaises(RuntimeError):
                    len(cache)
            cache.index([
                {'docno': '1', 'data': 'test'},
                {'docno': '2', 'data': 'caching pyterrier'},
                {'docno': '3', 'data': 'fetch me some data'},
                {'docno': '4', 'data': 'information retrieval'},
                {'docno': '5', 'data': 'foo bar baz'},
            ])
            with self.subTest('len when built'):
                self.assertEqual(len(cache), 5)
            with self.subTest('full iter'):
                self.assertEqual(list(cache), [
                    {'docno': '1', 'data': 'test'},
                    {'docno': '2', 'data': 'caching pyterrier'},
                    {'docno': '3', 'data': 'fetch me some data'},
                    {'docno': '4', 'data': 'information retrieval'},
                    {'docno': '5', 'data': 'foo bar baz'},
                ])
            with self.subTest('start only'):
                self.assertEqual(list(cache.get_corpus_iter(start=2)), [
                    {'docno': '3', 'data': 'fetch me some data'},
                    {'docno': '4', 'data': 'information retrieval'},
                    {'docno': '5', 'data': 'foo bar baz'},
                ])
            with self.subTest('stop only'):
                self.assertEqual(list(cache.get_corpus_iter(stop=2)), [
                    {'docno': '1', 'data': 'test'},
                    {'docno': '2', 'data': 'caching pyterrier'},
                ])
            with self.subTest('start and stop'):
                self.assertEqual(list(cache.get_corpus_iter(start=1, stop=3)), [
                    {'docno': '2', 'data': 'caching pyterrier'},
                    {'docno': '3', 'data': 'fetch me some data'},
                ])
            with self.subTest('empty'):
                self.assertEqual(list(cache.get_corpus_iter(start=3, stop=1)), [])
            with self.subTest('text_loader'):
                loader = cache.text_loader()
                pd.testing.assert_frame_equal(loader(pd.DataFrame([{'docno': '1'}])), pd.DataFrame([
                    {'docno': '1', 'data': 'test'},
                ]))
                pd.testing.assert_frame_equal(loader(pd.DataFrame([{'docno': '2'}])), pd.DataFrame([
                    {'docno': '2', 'data': 'caching pyterrier'},
                ]))
                pd.testing.assert_frame_equal(loader(pd.DataFrame([{'docno': '2'}, {'docno': '5'}, {'docno': '2'}])), pd.DataFrame([
                    {'docno': '2', 'data': 'caching pyterrier'},
                    {'docno': '5', 'data': 'foo bar baz'},
                    {'docno': '2', 'data': 'caching pyterrier'},
                ]))
                self.assertEqual(len(loader(pd.DataFrame([], columns=['docno']))), 0)

    def test_temp(self):
        with pyterrier_caching.IndexerCache() as cache:
            with self.subTest('len before built'):
                with self.assertRaises(RuntimeError):
                    len(cache)
            cache.index([
                {'docno': '1', 'data': 'test'},
                {'docno': '2', 'data': 'caching pyterrier'},
                {'docno': '3', 'data': 'fetch me some data'},
                {'docno': '4', 'data': 'information retrieval'},
                {'docno': '5', 'data': 'foo bar baz'},
            ])
            with self.subTest('len when built'):
                self.assertEqual(len(cache), 5)
            with self.subTest('full iter'):
                self.assertEqual(list(cache), [
                    {'docno': '1', 'data': 'test'},
                    {'docno': '2', 'data': 'caching pyterrier'},
                    {'docno': '3', 'data': 'fetch me some data'},
                    {'docno': '4', 'data': 'information retrieval'},
                    {'docno': '5', 'data': 'foo bar baz'},
                ])
            with self.subTest('start only'):
                self.assertEqual(list(cache.get_corpus_iter(start=2)), [
                    {'docno': '3', 'data': 'fetch me some data'},
                    {'docno': '4', 'data': 'information retrieval'},
                    {'docno': '5', 'data': 'foo bar baz'},
                ])
            with self.subTest('stop only'):
                self.assertEqual(list(cache.get_corpus_iter(stop=2)), [
                    {'docno': '1', 'data': 'test'},
                    {'docno': '2', 'data': 'caching pyterrier'},
                ])
            with self.subTest('start and stop'):
                self.assertEqual(list(cache.get_corpus_iter(start=1, stop=3)), [
                    {'docno': '2', 'data': 'caching pyterrier'},
                    {'docno': '3', 'data': 'fetch me some data'},
                ])
            with self.subTest('empty'):
                self.assertEqual(list(cache.get_corpus_iter(start=3, stop=1)), [])
            with self.subTest('text_loader'):
                loader = cache.text_loader()
                pd.testing.assert_frame_equal(loader(pd.DataFrame([{'docno': '1'}])), pd.DataFrame([
                    {'docno': '1', 'data': 'test'},
                ]))
                pd.testing.assert_frame_equal(loader(pd.DataFrame([{'docno': '2'}])), pd.DataFrame([
                    {'docno': '2', 'data': 'caching pyterrier'},
                ]))
                pd.testing.assert_frame_equal(loader(pd.DataFrame([{'docno': '2'}, {'docno': '5'}, {'docno': '2'}])), pd.DataFrame([
                    {'docno': '2', 'data': 'caching pyterrier'},
                    {'docno': '5', 'data': 'foo bar baz'},
                    {'docno': '2', 'data': 'caching pyterrier'},
                ]))
                self.assertEqual(len(loader(pd.DataFrame([], columns=['docno']))), 0)
        self.assertFalse(cache.built())

    def test_empty(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cache = pyterrier_caching.IndexerCache(d/'cache')
            cache.index(range(0))
            with self.subTest('len'):
                self.assertEqual(len(cache), 0)
            with self.subTest('iter'):
                self.assertEqual(list(cache), [])
