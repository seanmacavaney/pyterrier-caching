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
