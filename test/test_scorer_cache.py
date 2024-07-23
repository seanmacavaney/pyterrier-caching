import pandas as pd
import numpy as np
import tempfile
import unittest
from pathlib import Path
import pyterrier as pt
import pyterrier_caching
from npids import Lookup

class TestScorerCache(unittest.TestCase):
    def setUp(self):
        if not pt.started():
            pt.init()

    def test_build_from_iter(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cache = pyterrier_caching.ScorerCache(d/'cache')
            cache.build([
                {'docno': '1', 'data': 'test'},
                {'docno': '2', 'data': 'caching pyterrier'},
                {'docno': '3', 'data': 'fetch me some data'},
                {'docno': '4', 'data': 'information retrieval'},
                {'docno': '5', 'data': 'foo bar baz'},
            ])
            self.assertEqual(cache.corpus_count(), 5)

    def test_build_from_docnos(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cache = pyterrier_caching.ScorerCache(d/'cache')
            docnos = Lookup.build(['1', '2', '3', '4', '5'], d/'docnos.npids')
            cache.build(docnos_file=d/'docnos.npids')
            self.assertEqual(cache.corpus_count(), 5)

    def test_basic(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            scorer_1 = pt.apply.score(lambda df: 1.)
            scorer_raises = pt.apply.score(lambda df: 0/0)
            scorer_2 = pt.apply.score(lambda df: 2.)
            cache = pyterrier_caching.ScorerCache(d/'cache', scorer_1)
            docnos = Lookup.build(['1', '2', '3', '4', '5'], d/'docnos.npids')
            cache.build(docnos_file=d/'docnos.npids')
            # seeding the cache:
            res = cache(pd.DataFrame([
                {'qid': 'a', 'query': 'a', 'docno': '1'},
                {'qid': 'a', 'query': 'a', 'docno': '2'},
                {'qid': 'b', 'query': 'b', 'docno': '2'},
            ]))
            self.assertTrue((res == pd.DataFrame([
                {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 0},
                {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 1., 'rank': 1},
                {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1., 'rank': 0},
            ])).all().all())

            with self.subTest('make sure it doesn\'t call the scorer again on any of these'):
                cache.scorer = scorer_raises
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '2'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                ]))
                self.assertTrue((res == pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 0},
                    {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 1., 'rank': 1},
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1., 'rank': 0},
                ])).all().all())

            with self.subTest('make sure it calls a new scorer on any unknown values'):
                cache.scorer = scorer_2
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '3'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                    {'qid': 'b', 'query': 'a', 'docno': '1'}, # should pull from (a,1) cache (not dependent on qid)
                    {'qid': 'c', 'query': 'c', 'docno': '1'},
                ]))
                self.assertTrue((res == pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 1}, # old
                    {'qid': 'a', 'query': 'a', 'docno': '3', 'score': 2., 'rank': 0}, # new
                    {'qid': 'b', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 0}, # old
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1., 'rank': 1}, # old
                    {'qid': 'c', 'query': 'c', 'docno': '1', 'score': 2., 'rank': 0}, # new
                ])).all().all())


    def test_cached_retriever(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            scorer = pt.apply.score(lambda df: float(df['docno']))
            cache = pyterrier_caching.ScorerCache(d/'cache', scorer)
            docnos = Lookup.build(['1', '2', '3', '4', '5'], d/'docnos.npids')
            cache.build(docnos_file=d/'docnos.npids')
            # seeding the cache:
            res = cache(pd.DataFrame([
                {'qid': 'a', 'query': 'a', 'docno': '1'},
                {'qid': 'a', 'query': 'a', 'docno': '2'},
                {'qid': 'b', 'query': 'b', 'docno': '2'},
            ]))

            with self.subTest('should raise error when retrieving, since not all docs are cached'):
                with self.assertRaises(RuntimeError):
                    cache.cached_retriever()(pd.DataFrame([
                        {'qid': 'a', 'query': 'a'},
                        {'qid': 'b', 'query': 'b'},
                    ]))

            # completely score query a
            res = cache(pd.DataFrame([
                {'qid': 'a', 'query': 'a', 'docno': '3'},
                {'qid': 'a', 'query': 'a', 'docno': '4'},
                {'qid': 'a', 'query': 'a', 'docno': '5'},
            ]))

            with self.subTest('query a should return cached values'):
                res = cache.cached_retriever(num_results=2)(pd.DataFrame([
                    {'qid': 'a', 'query': 'a'},
                ]))
                self.assertTrue((res == pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '5', 'score': 5., 'rank': 0},
                    {'qid': 'a', 'query': 'a', 'docno': '4', 'score': 4., 'rank': 1},
                ])).all().all())

            with self.subTest('num_results is robust'):
                res = cache.cached_retriever(num_results=1000)(pd.DataFrame([
                    {'qid': 'a', 'query': 'a'},
                ]))
                self.assertTrue((res == pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '5', 'score': 5., 'rank': 0},
                    {'qid': 'a', 'query': 'a', 'docno': '4', 'score': 4., 'rank': 1},
                    {'qid': 'a', 'query': 'a', 'docno': '3', 'score': 3., 'rank': 2},
                    {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 2., 'rank': 3},
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 4},
                ])).all().all())

            with self.subTest('query b should still raise an error'):
                with self.assertRaises(RuntimeError):
                    cache.cached_retriever(num_results=2)(pd.DataFrame([
                        {'qid': 'b', 'query': 'b'},
                    ]))

            with self.subTest('should still raise an error when retrieving both'):
                with self.assertRaises(RuntimeError):
                    cache.cached_retriever(num_results=2)(pd.DataFrame([
                        {'qid': 'a', 'query': 'a'},
                        {'qid': 'b', 'query': 'b'},
                    ]))

            with self.subTest('should raise for an unseen query'):
                with self.assertRaises(RuntimeError):
                    cache.cached_retriever(num_results=2)(pd.DataFrame([
                        {'qid': 'c', 'query': 'c'},
                    ]))
