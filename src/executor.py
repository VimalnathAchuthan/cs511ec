'''
    Author: University of Illinois at Urbana Champaign
    DateTime: 2023-11-05 10:42:47
    FilePath: src/pandas_q4.py
    Description:
'''

import pandas as pd
import numpy as np
import os

from src.difference import Difference
from src.optimizer import Optimizer
from src.report import get_file, report


class Executor(object):
    def __init__(self, data: str, pattern: str):
        self._opt = Optimizer(data, pattern)
        self._opt.run()
        self._plan = self._opt._plan
        self._data = self._opt._data
        self._pattern = self._opt._pattern

    @report
    def result(self) -> str:
        df_G = self._data.df
        df_P = self._pattern.df

        pattern_vertices = np.sort(pd.unique(df_P[['src_id', 'dst_id']].values.ravel()))
        variable_names = ['u{}'.format(i+1) for i in range(len(pattern_vertices))]
        dict_pv = dict(zip(pattern_vertices, variable_names))
        dict_pv_rev = {v: k for k, v in dict_pv.items()}

        relations = {}

        for item, _ in self._plan:
            ui, uj = item.split(',')

            src_id_p = dict_pv_rev[ui]
            dst_id_p = dict_pv_rev[uj]

            edge_p = df_P[
                ((df_P['src_id'] == src_id_p) & (df_P['dst_id'] == dst_id_p)) |
                ((df_P['src_id'] == dst_id_p) & (df_P['dst_id'] == src_id_p))
            ].iloc[0]

            src_label_p = edge_p['src_label']
            dst_label_p = edge_p['dst_label']
            edge_label_p = edge_p['edge_label']
            edge_type_p = edge_p['edge_type']

            if edge_type_p == 0:  # Directed
                df_match = df_G[
                    (df_G['edge_type'] == 0) &
                    (df_G['edge_label'] == edge_label_p) &
                    (df_G['src_label'] == src_label_p) &
                    (df_G['dst_label'] == dst_label_p)
                ][['src_id', 'dst_id']].rename(columns={'src_id': ui, 'dst_id': uj})
            else:  # Undirected
                df_match = df_G[
                    (df_G['edge_type'] == 1) &
                    (df_G['edge_label'] == edge_label_p) &
                    (
                        ((df_G['src_label'] == src_label_p) & (df_G['dst_label'] == dst_label_p)) |
                        ((df_G['src_label'] == dst_label_p) & (df_G['dst_label'] == src_label_p))
                    )
                ][['src_id', 'dst_id']]
                df_match_rev = df_match.rename(columns={'src_id': uj, 'dst_id': ui})
                df_match = df_match.rename(columns={'src_id': ui, 'dst_id': uj})
                df_match = pd.concat([df_match, df_match_rev], ignore_index=True)

            df_match.drop_duplicates(inplace=True)
            relations[item] = df_match

        # Execute joins according to the plan
        result_df = None
        for item, _ in self._plan:
            df_relation = relations[item]
            if result_df is None:
                result_df = df_relation
            else:
                common_vars = list(set(result_df.columns) & set(df_relation.columns))
                result_df = pd.merge(result_df, df_relation, how='inner', on=common_vars, sort=False)

        # Ensure all variables are present
        for var in variable_names:
            if var not in result_df.columns:
                result_df[var] = np.nan

        # Finalize the result
        result_df = result_df[variable_names]
        result_df.dropna(inplace=True)
        result_df = result_df.astype(np.int32)
        result_df.sort_values(by=variable_names, inplace=True)

        output_file = os.path.join('out', 'executor.csv')
        os.makedirs('out', exist_ok=True)
        result_df.to_csv(output_file, index=False, header=False)

        return output_file


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
