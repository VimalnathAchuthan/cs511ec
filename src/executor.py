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
        df_G = self._data.df.copy()
        df_P = self._pattern.df.copy()

        df_G['min_id'] = df_G[['src_id', 'dst_id']].min(axis=1)
        df_G['max_id'] = df_G[['src_id', 'dst_id']].max(axis=1)
        df_G['edge_key'] = df_G['min_id'].astype(str) + '_' + df_G['max_id'].astype(str) + '_' + df_G['edge_label'].astype(str)

        df_G_counts = df_G.groupby('edge_key').size().reset_index(name='count')
        df_G = df_G.merge(df_G_counts, on='edge_key', how='left')

        df_G['edge_type'] = df_G['count'].apply(lambda x: 'undirected' if x == 2 else 'directed')

        df_G.set_index(['edge_type', 'edge_label', 'src_label', 'dst_label'], inplace=True)

        df_P['min_id'] = df_P[['src_id', 'dst_id']].min(axis=1)
        df_P['max_id'] = df_P[['src_id', 'dst_id']].max(axis=1)
        df_P['edge_key'] = df_P['min_id'].astype(str) + '_' + df_P['max_id'].astype(str) + '_' + df_P['edge_label'].astype(str)
        df_P_counts = df_P.groupby('edge_key').size().reset_index(name='count')
        df_P = df_P.merge(df_P_counts, on='edge_key', how='left')
        df_P['edge_type'] = df_P['count'].apply(lambda x: 'undirected' if x == 2 else 'directed')

        pattern_vertices = pd.unique(df_P[['src_id', 'dst_id']].values.ravel('K'))
        pattern_vertices.sort()
        variable_names = ['u{}'.format(i+1) for i in range(len(pattern_vertices))]
        dict_pv = dict(zip(pattern_vertices, variable_names))
        dict_pv_rev = {v: k for k, v in dict_pv.items()}

        relations = {}

        for item, _ in self._plan:
            ui, uj = item.split(',')

            src_id_p = dict_pv_rev[ui]
            dst_id_p = dict_pv_rev[uj]

            edge_p = df_P[
                ((df_P['src_id'] == src_id_p) & (df_P['dst_id'] == dst_id_p) & (df_P['edge_label'].notnull())) |
                ((df_P['src_id'] == dst_id_p) & (df_P['dst_id'] == src_id_p) & (df_P['edge_label'].notnull()))
            ].iloc[0]

            src_label_p = edge_p['src_label']
            dst_label_p = edge_p['dst_label']
            edge_label_p = edge_p['edge_label']
            edge_type_p = edge_p['edge_type']

            if edge_type_p == 'directed':
                condition = (
                    (df_G.index.get_level_values('edge_type') == 'directed') &
                    (df_G.index.get_level_values('edge_label') == edge_label_p) &
                    (df_G.index.get_level_values('src_label') == src_label_p) &
                    (df_G.index.get_level_values('dst_label') == dst_label_p)
                )
                df_match = df_G[condition].reset_index()[['src_id', 'dst_id']]
                df_match.columns = [ui, uj]
            else:
                condition1 = (
                    (df_G.index.get_level_values('edge_type') == 'undirected') &
                    (df_G.index.get_level_values('edge_label') == edge_label_p) &
                    (df_G.index.get_level_values('src_label') == src_label_p) &
                    (df_G.index.get_level_values('dst_label') == dst_label_p)
                )
                condition2 = (
                    (df_G.index.get_level_values('edge_type') == 'undirected') &
                    (df_G.index.get_level_values('edge_label') == edge_label_p) &
                    (df_G.index.get_level_values('src_label') == dst_label_p) &
                    (df_G.index.get_level_values('dst_label') == src_label_p)
                )
                df_match1 = df_G[condition1].reset_index()[['src_id', 'dst_id']]
                df_match2 = df_G[condition2].reset_index()[['src_id', 'dst_id']]
                df_match1.columns = [ui, uj]
                df_match2.columns = [ui, uj]
                df_match = pd.concat([df_match1, df_match2], ignore_index=True)

            df_match.drop_duplicates(inplace=True)
            relations[item] = df_match

        result_df = None
        for item, _ in self._plan:
            df_relation = relations[item]
            if result_df is None:
                result_df = df_relation
            else:
                result_df = pd.merge(result_df, df_relation, how='inner')

        if result_df is not None and not result_df.empty:
            for var in variable_names:
                if var not in result_df.columns:
                    result_df[var] = np.nan

            result_df = result_df[variable_names]
            result_df.dropna(inplace=True)
            result_df = result_df.astype(int)
            result_df.sort_values(by=variable_names, inplace=True)
        else:
            result_df = pd.DataFrame(columns=variable_names)

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
