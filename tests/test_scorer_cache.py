import numpy as np
import pandas as pd
import tempfile
import unittest
from pathlib import Path
import pyterrier as pt
import pyterrier_caching
from npids import Lookup


def raises(df):
    raise RuntimeError()


class TestDenseScorerCache(unittest.TestCase):
    def test_build_from_iter(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cache = pyterrier_caching.DenseScorerCache(d/'cache')
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
            cache = pyterrier_caching.DenseScorerCache(d/'cache')
            docnos = Lookup.build(['1', '2', '3', '4', '5'], d/'docnos.npids')
            cache.build(docnos_file=d/'docnos.npids')
            self.assertEqual(cache.corpus_count(), 5)

    def test_basic(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            scorer_1 = pt.apply.score(lambda df: 1.)
            scorer_raises = pt.apply.score(raises)
            scorer_2 = pt.apply.score(lambda df: 2.)
            cache = pyterrier_caching.DenseScorerCache(d/'cache', scorer_1)
            docnos = Lookup.build(['1', '2', '3', '4', '5'], d/'docnos.npids')
            cache.build(docnos_file=d/'docnos.npids')
            # seeding the cache:
            res = cache(pd.DataFrame([
                {'qid': 'a', 'query': 'a', 'docno': '1'},
                {'qid': 'a', 'query': 'a', 'docno': '2'},
                {'qid': 'b', 'query': 'b', 'docno': '2'},
            ]))
            pd.testing.assert_frame_equal(res, pd.DataFrame([
                {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 0},
                {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 1., 'rank': 1},
                {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1., 'rank': 0},
            ]))

            with self.subTest('make sure it doesn\'t call the scorer again on any of these'):
                cache.scorer = scorer_raises
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '2'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                ]))
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 0},
                    {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 1., 'rank': 1},
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1., 'rank': 0},
                ]))

            with self.subTest('make sure it calls a new scorer on any unknown values'):
                cache.scorer = scorer_2
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '3'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                    {'qid': 'b', 'query': 'a', 'docno': '1'}, # should pull from (a,1) cache (not dependent on qid)
                    {'qid': 'c', 'query': 'c', 'docno': '1'},
                ]))
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 1}, # old
                    {'qid': 'a', 'query': 'a', 'docno': '3', 'score': 2., 'rank': 0}, # new
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1., 'rank': 0}, # old
                    {'qid': 'b', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 1}, # old
                    {'qid': 'c', 'query': 'c', 'docno': '1', 'score': 2., 'rank': 0}, # new
                ]))

            with self.subTest('reload the cache and make sure we get the same results'):
                cache = pyterrier_caching.DenseScorerCache(d/'cache', scorer_2)
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '3'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                    {'qid': 'b', 'query': 'a', 'docno': '1'}, # should pull from (a,1) cache (not dependent on qid)
                    {'qid': 'c', 'query': 'c', 'docno': '1'},
                ]))
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 1}, # old
                    {'qid': 'a', 'query': 'a', 'docno': '3', 'score': 2., 'rank': 0}, # new
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1., 'rank': 0}, # old
                    {'qid': 'b', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 1}, # old
                    {'qid': 'c', 'query': 'c', 'docno': '1', 'score': 2., 'rank': 0}, # new
                ]))

            with self.subTest('no scorer necessary if everything is already in the cache'):
                cache = pyterrier_caching.DenseScorerCache(d/'cache')
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '3'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                    {'qid': 'b', 'query': 'a', 'docno': '1'}, # should pull from (a,1) cache (not dependent on qid)
                    {'qid': 'c', 'query': 'c', 'docno': '1'},
                ]))
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 1}, # old
                    {'qid': 'a', 'query': 'a', 'docno': '3', 'score': 2., 'rank': 0}, # new
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1., 'rank': 0}, # old
                    {'qid': 'b', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 1}, # old
                    {'qid': 'c', 'query': 'c', 'docno': '1', 'score': 2., 'rank': 0}, # new
                ]))

            with self.subTest('raises error if document is requested that wasn\'t in the cache and no scorer is provided'):
                with self.assertRaises(LookupError):
                    cache(pd.DataFrame([
                        {'qid': 'a', 'query': 'a', 'docno': '4'},
                        {'qid': 'a', 'query': 'a', 'docno': '3'},
                        {'qid': 'b', 'query': 'b', 'docno': '2'},
                        {'qid': 'b', 'query': 'a', 'docno': '1'},
                        {'qid': 'c', 'query': 'c', 'docno': '1'},
                    ]))

    def test_cached_retriever(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            scorer = pt.apply.score(lambda df: float(df['docno']))
            cache = pyterrier_caching.DenseScorerCache(d/'cache', scorer)
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
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '5', 'score': 5., 'rank': 0},
                    {'qid': 'a', 'query': 'a', 'docno': '4', 'score': 4., 'rank': 1},
                ]))

            with self.subTest('num_results is robust'):
                res = cache.cached_retriever(num_results=1000)(pd.DataFrame([
                    {'qid': 'a', 'query': 'a'},
                ]))
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '5', 'score': 5., 'rank': 0},
                    {'qid': 'a', 'query': 'a', 'docno': '4', 'score': 4., 'rank': 1},
                    {'qid': 'a', 'query': 'a', 'docno': '3', 'score': 3., 'rank': 2},
                    {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 2., 'rank': 3},
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 4},
                ]))

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


class TestSparseScorerCache(unittest.TestCase):
    def test_basic(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            scorer_1 = pt.apply.score(lambda df: 1.)
            scorer_raises = pt.apply.score(raises)
            scorer_2 = pt.apply.score(lambda df: 2.)
            cache = pyterrier_caching.SparseScorerCache(d/'cache', scorer_1)
            # seeding the cache:
            res = cache(pd.DataFrame([
                {'qid': 'a', 'query': 'a', 'docno': '1'},
                {'qid': 'a', 'query': 'a', 'docno': '2'},
                {'qid': 'b', 'query': 'b', 'docno': '2'},
            ]))
            pd.testing.assert_frame_equal(res, pd.DataFrame([
                {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 0},
                {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 1., 'rank': 1},
                {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1., 'rank': 0},
            ]))

            with self.subTest('make sure it doesn\'t call the scorer again on any of these'):
                cache.scorer = scorer_raises
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '2'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                ]))
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 0},
                    {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 1., 'rank': 1},
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1., 'rank': 0},
                ]))

            with self.subTest('make sure it calls a new scorer on any unknown values'):
                cache.scorer = scorer_2
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '3'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                    {'qid': 'b', 'query': 'a', 'docno': '1'}, # should pull from (a,1) cache (not dependent on qid)
                    {'qid': 'c', 'query': 'c', 'docno': '1'},
                ]))
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 1}, # old
                    {'qid': 'a', 'query': 'a', 'docno': '3', 'score': 2., 'rank': 0}, # new
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1., 'rank': 0}, # old
                    {'qid': 'b', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 1}, # old
                    {'qid': 'c', 'query': 'c', 'docno': '1', 'score': 2., 'rank': 0}, # new
                ]))

            cache = None
            with self.subTest('reload the cache and make sure we get the same results'):
                cache = pyterrier_caching.SparseScorerCache(d/'cache', scorer_2)
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '3'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                    {'qid': 'b', 'query': 'a', 'docno': '1'}, # should pull from (a,1) cache (not dependent on qid)
                    {'qid': 'c', 'query': 'c', 'docno': '1'},
                ]))
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 1}, # old
                    {'qid': 'a', 'query': 'a', 'docno': '3', 'score': 2., 'rank': 0}, # new
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1., 'rank': 0}, # old
                    {'qid': 'b', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 1}, # old
                    {'qid': 'c', 'query': 'c', 'docno': '1', 'score': 2., 'rank': 0}, # new
                ]))

            with self.subTest('no scorer necessary if everything is already in the cache'):
                cache = pyterrier_caching.SparseScorerCache(d/'cache')
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '3'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                    {'qid': 'b', 'query': 'a', 'docno': '1'}, # should pull from (a,1) cache (not dependent on qid)
                    {'qid': 'c', 'query': 'c', 'docno': '1'},
                ]))
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 1}, # old
                    {'qid': 'a', 'query': 'a', 'docno': '3', 'score': 2., 'rank': 0}, # new
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1., 'rank': 0}, # old
                    {'qid': 'b', 'query': 'a', 'docno': '1', 'score': 1., 'rank': 1}, # old
                    {'qid': 'c', 'query': 'c', 'docno': '1', 'score': 2., 'rank': 0}, # new
                ]))

            with self.subTest('raises error if document is requested that wasn\'t in the cache and no scorer is provided'):
                with self.assertRaises(LookupError):
                    cache(pd.DataFrame([
                        {'qid': 'a', 'query': 'a', 'docno': '4'},
                        {'qid': 'a', 'query': 'a', 'docno': '3'},
                        {'qid': 'b', 'query': 'b', 'docno': '2'},
                        {'qid': 'b', 'query': 'a', 'docno': '1'},
                        {'qid': 'c', 'query': 'c', 'docno': '1'},
                    ]))
    def test_features(self):
        with tempfile.TemporaryDirectory() as d:
            feat_1 = pt.apply.doc_features(lambda row: np.array([0.2, 0.1]) if row["qid"] == "q1" else  np.array([0.4, 0.3]))
            d = Path(d)
            cache = pyterrier_caching.SparseScorerCache(d/'cache', feat_1, value="features", pickle=True)
            
            df1 = pd.DataFrame([{'qid' : 'q1', 'query' : 'a', 'docno' : 'd1'}])
            res = cache.transform(df1)
            res1c = cache.transform(df1)
            self.assertEqual(0.2, res1c.iloc[0]["features"][0])
            pd.testing.assert_frame_equal(res, res1c)

            df2 = pd.DataFrame([{'qid' : 'q2', 'query' : 'ab', 'docno' : 'd1'}])
            res = cache.transform(df2)
            res1c = cache.transform(df2)
            self.assertEqual(0.4, res1c.iloc[0]["features"][0])
            pd.testing.assert_frame_equal(res, res1c)
        
