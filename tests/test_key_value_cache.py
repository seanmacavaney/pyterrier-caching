import pandas as pd
import tempfile
import unittest
from pathlib import Path
import pyterrier as pt
import pyterrier_caching


def raises(df):
    raise RuntimeError()


class TestSparseScorerCache(unittest.TestCase):
    def test_basic(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            scorer_1 = pt.apply.score(lambda df: 1.)
            scorer_raises = pt.apply.score(raises)
            scorer_2 = pt.apply.score(lambda df: 2.)
            cache = pyterrier_caching.KeyValueCache(d/'cache', scorer_1, key=['query', 'docno'], value=['score'])
            # seeding the cache:
            res = cache(pd.DataFrame([
                {'qid': 'a', 'query': 'a', 'docno': '1'},
                {'qid': 'a', 'query': 'a', 'docno': '2'},
                {'qid': 'b', 'query': 'b', 'docno': '2'},
            ]))
            pd.testing.assert_frame_equal(res, pd.DataFrame([
                {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1.},
                {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 1.},
                {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1.},
            ]))

            with self.subTest('make sure it doesn\'t call the transformer again on any of these'):
                cache.transformer = scorer_raises
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '2'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                ]))
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1.},
                    {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 1.},
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1.},
                ]))

            with self.subTest('make sure it calls a new transformer on any unknown values'):
                cache.transformer = scorer_2
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '3'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                    {'qid': 'b', 'query': 'a', 'docno': '1'}, # should pull from (a,1) cache (not dependent on qid)
                    {'qid': 'c', 'query': 'c', 'docno': '1'},
                ]))
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1.}, # old
                    {'qid': 'a', 'query': 'a', 'docno': '3', 'score': 2.}, # new
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1.}, # old
                    {'qid': 'b', 'query': 'a', 'docno': '1', 'score': 1.}, # old
                    {'qid': 'c', 'query': 'c', 'docno': '1', 'score': 2.}, # new
                ]))

            cache = None
            with self.subTest('reload the cache and make sure we get the same results'):
                cache = pyterrier_caching.KeyValueCache(d/'cache', scorer_2)
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '3'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                    {'qid': 'b', 'query': 'a', 'docno': '1'}, # should pull from (a,1) cache (not dependent on qid)
                    {'qid': 'c', 'query': 'c', 'docno': '1'},
                ]))
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1.}, # old
                    {'qid': 'a', 'query': 'a', 'docno': '3', 'score': 2.}, # new
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1.}, # old
                    {'qid': 'b', 'query': 'a', 'docno': '1', 'score': 1.}, # old
                    {'qid': 'c', 'query': 'c', 'docno': '1', 'score': 2.}, # new
                ]))

            with self.subTest('no scorer necessary if everything is already in the cache'):
                cache = pyterrier_caching.KeyValueCache(d/'cache')
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1'},
                    {'qid': 'a', 'query': 'a', 'docno': '3'},
                    {'qid': 'b', 'query': 'b', 'docno': '2'},
                    {'qid': 'b', 'query': 'a', 'docno': '1'}, # should pull from (a,1) cache (not dependent on qid)
                    {'qid': 'c', 'query': 'c', 'docno': '1'},
                ]))
                pd.testing.assert_frame_equal(res, pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 1.}, # old
                    {'qid': 'a', 'query': 'a', 'docno': '3', 'score': 2.}, # new
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 1.}, # old
                    {'qid': 'b', 'query': 'a', 'docno': '1', 'score': 1.}, # old
                    {'qid': 'c', 'query': 'c', 'docno': '1', 'score': 2.}, # new
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
