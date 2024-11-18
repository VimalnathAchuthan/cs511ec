'''
    Author: University of Illinois at Urbana Champaign
    DateTime: 2023-11-05 10:42:47
    FilePath: src/pandas_q1.py
    Description:
'''

import pandas as pd

from src.difference import Difference
from src.report import get_file, report


@report
def pandas_q1(data_file: str, pattern_file: str) -> str:
    data_columns = ['src', 'dst', 'src_label', 'dst_label', 'edge_label']
    pattern_columns = ['p_src', 'p_dst', 'p_src_label', 'p_dst_label', 'p_edge_label']

    data = pd.read_csv(data_file, sep=" ", header=None, names=data_columns)
    pattern = pd.read_csv(pattern_file, sep=" ", header=None, names=pattern_columns)

    matched_edges = data.merge(
        pattern,
        left_on=['src_label', 'dst_label', 'edge_label'],
        right_on=['p_src_label', 'p_dst_label', 'p_edge_label']
    )

    results = matched_edges[['src', 'dst', 'p_src', 'p_dst']]
    results = results.rename(columns={'src': 'u1', 'dst': 'u2', 'p_src': 'p1', 'p_dst': 'p2'})

    results = results.sort_values(by=['p1', 'p2', 'u1', 'u2'])

    output_file = 'out/pandas_q1.csv'
    results.to_csv(output_file, index=False, header=False)

    return output_file


def test(data, pattern, expect) -> int:
    # import the logger to output message
    import logging
    logger = logging.getLogger()
    data = get_file('data', data)
    pattern = get_file('pattern', pattern)
    expect = get_file('expect', expect)

    # run the test
    print("**************begin pandas_q1 test**************")
    diff = Difference(pandas_q1(data, pattern), expect)
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
    pattern = 'pattern/1.txt'
    expect = 'expect/d1-p1.txt'
    test(data, pattern, expect)
