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
    data = pd.read_csv(data_file, sep=" ", header=None, 
                      names=['v1', 'v2', 'l1', 'l2', 'e'])
    pattern = pd.read_csv(pattern_file, sep=" ", header=None,
                         names=['v1', 'v2', 'l1', 'l2', 'e'])

    edges = {}
    pattern_edges = {}
    
    for _, row in pattern.iterrows():
        edge_key = f"u{row['v1']+1},u{row['v2']+1}"
        pattern_edges[edge_key] = (row['l1'], row['l2'], row['e'])

    for pattern_key, (l1, l2, e) in pattern_edges.items():
        matches = data[
            (data['l1'] == l1) & 
            (data['l2'] == l2) & 
            (data['e'] == e)
        ][['v1', 'v2']]
        edges[pattern_key] = matches.rename(
            columns={'v1': pattern_key.split(',')[0], 
                    'v2': pattern_key.split(',')[1]}
        )

    result = None
    for edge_relation in edges.values():
        if result is None:
            result = edge_relation
        else:
            result = pd.merge(result, edge_relation)

    result = result.sort_values(by=list(result.columns))

    output_file = 'out/pandas_q1.csv'
    result.to_csv(output_file, index=False, header=False)
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
