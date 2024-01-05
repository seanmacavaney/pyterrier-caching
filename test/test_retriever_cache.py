import pandas as pd
import numpy as np
import tempfile
import unittest
from pathlib import Path
import pyterrier as pt
import pyterrier_caching
from npids import Lookup

class TestRetrieverCache(unittest.TestCase):
    def setUp(self):
        if not pt.started():
            pt.init()

    def test_basic(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            def retriever_1(df):
                res = pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 2., 'rank': 0},
                    {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 1., 'rank': 1},
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 3., 'rank': 0},
                ])
                return res[res['query'].isin(df['query'])]
            def retriever_2(df):
                res = pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '3', 'score': 3., 'rank': 0},
                    {'qid': 'a', 'query': 'a', 'docno': '4', 'score': 2., 'rank': 1},
                    {'qid': 'c', 'query': 'c', 'docno': '0', 'score': 0., 'rank': 0},
                ])
                return res[res['query'].isin(df['query'])]
            retriever_raises = pt.apply.generic(lambda df: 0/0)
            # retriever_2 = pt.apply.generic(lambda df: 2.)
            cache = pyterrier_caching.RetrieverCache(d/'cache', pt.apply.generic(retriever_1))
            # seeding the cache:
            res = cache(pd.DataFrame([
                {'qid': 'a', 'query': 'a'},
                {'qid': 'b', 'query': 'b'},
            ]))
            self.assertTrue((res == pd.DataFrame([
                {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 2., 'rank': 0},
                {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 1., 'rank': 1},
                {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 3., 'rank': 0},
            ])).all().all())

            with self.subTest('make sure it doesn\'t call the retriever again on any of these'):
                cache.retriever = retriever_raises
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a'},
                    {'qid': 'b', 'query': 'b'},
                ]))
                self.assertTrue((res == pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 2., 'rank': 0},
                    {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 1., 'rank': 1},
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 3., 'rank': 0},
                ])).all().all())

            with self.subTest('only some of the queries'):
                cache.retriever = retriever_raises
                res = cache(pd.DataFrame([
                    {'qid': 'b', 'query': 'b'},
                ]))
                self.assertTrue((res == pd.DataFrame([
                    {'qid': 'b', 'query': 'b', 'docno': '2', 'score': 3., 'rank': 0},
                ])).all().all())

            with self.subTest('make sure it calls a new scorer on any unknown values'):
                cache.retriever = pt.apply.generic(retriever_2)
                res = cache(pd.DataFrame([
                    {'qid': 'a', 'query': 'a'},
                    {'qid': 'c', 'query': 'c'},
                ]))
                self.assertTrue((res == pd.DataFrame([
                    {'qid': 'a', 'query': 'a', 'docno': '1', 'score': 2., 'rank': 0}, # old
                    {'qid': 'a', 'query': 'a', 'docno': '2', 'score': 1., 'rank': 1}, # old
                    {'qid': 'c', 'query': 'c', 'docno': '0', 'score': 0., 'rank': 0}, # new
                ])).all().all())
