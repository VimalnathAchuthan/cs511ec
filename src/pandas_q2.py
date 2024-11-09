'''
    Author: University of Illinois at Urbana Champaign
    DateTime: 2023-11-05 10:42:47
    FilePath: src/pandas_q2.py
    Description:
'''

import pandas as pd

from src.difference import Difference
from src.report import get_file, report


@report
def pandas_q2(data_file: str, pattern_file: str) -> str:
    # TODO: begin of your codes
    return 'out/pandas_q2.csv'
    # TODO: end of your codes


def test(data, pattern, expect) -> int:
    # import the logger to output message
    import logging
    logger = logging.getLogger()
    data = get_file('data', data)
    pattern = get_file('pattern', pattern)
    expect = get_file('expect', expect)


    # run the test
    print("**************begin pandas_q2 test**************")
    diff = Difference(pandas_q2(data, pattern), expect)
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
    data = 'data/1.txt'
    pattern = 'pattern/2.txt'
    expect = 'expect/d1-p2.txt'
    test(data, pattern, expect)
