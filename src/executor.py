'''
    Author: University of Illinois at Urbana Champaign
    DateTime: 2023-11-05 10:42:47
    FilePath: src/pandas_q4.py
    Description:
'''

import pandas as pd

from src.difference import Difference
from src.optimizer import Optimizer
from src.report import get_file, report


class Executor(object):
    def __init__(self, data: str, pattern: str):
        self._opt = Optimizer(data, pattern)
        # TODO: begin of your codes
        '''
        requirement:
            1. [optional] can define and initialize new fields here
            2. please put expensive computation inside self.result()
        '''
        # TODO: end of your codes

    @report
    def result(self) -> str:
        # TODO: begin of your codes
        '''
        requirements
            1. use the Optimizer to compute an optimized plan
            2. follow the plan to compute the result
            3. save the final result to a csv file
            4. return a string pointing the output file
            5. you can add new fields and methods, and use them here
        '''
        return 'out/executor.csv'
        # TODO: end of your codes


def test_count(data: str, pattern: str, row_e: int, column_e: int) -> int:
    # import the logger to output message
    import logging
    logger = logging.getLogger()
    data = get_file('data', data)
    pattern = get_file('pattern', pattern)

    # run the test
    print("**************begin Executor test_count**************")
    actual = Executor(data, pattern).result()
    max_col = -1
    min_col = -1
    row_a = 0
    with open(actual, 'r', encoding='utf-8') as r:
        for line in r:
            row = line.strip().split(',')
            if max_col < 0:
                max_col = len(row)
            else:
                max_col = max(max_col, len(row))
            if min_col < 0:
                min_col = len(row)
            else:
                min_col = min(min_col, len(row))
            row_a += 1
    try:
        assert(row_a == row_e)
        assert(min_col == max_col)
        assert(min_col == column_e)
        print('*******************pass*******************')
        return 30
    except Exception as e:
        logger.error("Exception Occurred:" + str(e))
        print('*******************fail*******************')
        col_a = (min_col + max_col) / 2.0
        print(f'actual dimension: {row_a}, {col_a:.1f}')
        print(f'expect dimension: {row_e}, {column_e}')
        return 0


def test_match(data: str, pattern: str, expect: str) -> int:
    # import the logger to output message
    import logging
    logger = logging.getLogger()
    data = get_file('data', data)
    pattern = get_file('pattern', pattern)
    expect = get_file('expect', expect)

    # run the test
    print("**************begin Executor test_match**************")
    exe = Executor(data, pattern)
    diff = Difference(exe.result(), expect)
    try:
        assert(diff.match)
        print('*******************pass*******************')
        return 20
    except Exception as e:
        logger.error("Exception Occurred:" + str(e))
        print('*******************fail*******************')
        print(diff.actual)
        print(diff.expect)
        return 0


if __name__ == "__main__":
    data = 'data/2.txt'
    pattern = 'pattern/3.txt'
    expect = 'expect/d2-p3.txt'
    test_match(data, pattern, expect)
