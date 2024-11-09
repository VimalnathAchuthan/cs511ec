'''
    Author: University of Illinois at Urbana Champaign
    DateTime: 2023-11-05 21:15:58
    FilePath: src/difference.py
    Description:
'''


import itertools


class Difference(object):
    def __init__(self, actual: str, expect: str):
        self._match = False
        with open(actual, 'r') as ra, open(expect, 'r') as re:
            for line_a, line_e in itertools.zip_longest(ra, re, fillvalue=''):
                line_a = line_a.strip()
                line_e = line_e.strip()
                row_a = line_a.split(',')
                row_e = line_e.split()
                if len(row_a) != len(row_e):
                    self._actual = f'actual column size {len(row_a)}'
                    self._expect = f'expect column size {len(row_e)}'
                    return
                for item_a, item_e in zip(row_a, row_e):
                    if item_a != item_e:
                        self._actual = f'actual result {line_a}'
                        self._expect = f'expect result {line_e}'
                        return
        self._match = True

    @property
    def actual(self) -> str:
        return self._actual

    @property
    def expect(self) -> str:
        return self._expect

    @property
    def match(self) -> bool:
        return self._match
