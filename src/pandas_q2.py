'''
    Author: University of Illinois at Urbana Champaign
    DateTime: 2023-11-05 10:42:47
    FilePath: src/pandas_q2.py
    Description:
'''

import pandas as pd
import os

from src.difference import Difference
from src.report import get_file, report


@report
def pandas_q2(data_file: str, pattern_file: str) -> str:
    df_G = pd.read_csv(data_file, sep=' ', header=None, names=['src_id', 'dst_id', 'src_label', 'dst_label', 'edge_label'])

    df_P = pd.read_csv(pattern_file, sep=' ', header=None, names=['src_id_p', 'dst_id_p', 'src_label_p', 'dst_label_p', 'edge_label_p'])

    df_G['min_id'] = df_G[['src_id', 'dst_id']].min(axis=1)
    df_G['max_id'] = df_G[['src_id', 'dst_id']].max(axis=1)
    df_G['edge_key'] = df_G['min_id'].astype(str) + '_' + df_G['max_id'].astype(str) + '_' + df_G['edge_label'].astype(str)

    df_G_counts = df_G.groupby('edge_key').size().reset_index(name='count')

    df_G = df_G.merge(df_G_counts, on='edge_key', how='left')

    df_G['edge_type'] = df_G['count'].apply(lambda x: 'undirected' if x == 2 else 'directed')

    df_P['min_id'] = df_P[['src_id_p', 'dst_id_p']].min(axis=1)
    df_P['max_id'] = df_P[['src_id_p', 'dst_id_p']].max(axis=1)
    df_P['edge_key'] = df_P['min_id'].astype(str) + '_' + df_P['max_id'].astype(str) + '_' + df_P['edge_label_p'].astype(str)

    df_P_counts = df_P.groupby('edge_key').size().reset_index(name='count')
    df_P = df_P.merge(df_P_counts, on='edge_key', how='left')
    df_P['edge_type'] = df_P['count'].apply(lambda x: 'undirected' if x == 2 else 'directed')

    pattern_vertices = pd.unique(df_P[['src_id_p', 'dst_id_p']].values.ravel('K'))
    pattern_vertices.sort()
    variable_names = ['u{}'.format(i+1) for i in range(len(pattern_vertices))]
    dict_pv = dict(zip(pattern_vertices, variable_names))

    R_list = []

    df_P_unique = df_P.drop_duplicates(subset=['edge_key', 'edge_type'])
    for idx, edge_p in df_P_unique.iterrows():
        src_id_p = edge_p['src_id_p']
        dst_id_p = edge_p['dst_id_p']
        src_label_p = edge_p['src_label_p']
        dst_label_p = edge_p['dst_label_p']
        edge_label_p = edge_p['edge_label_p']
        edge_type_p = edge_p['edge_type']

        ui = dict_pv[src_id_p]
        uj = dict_pv[dst_id_p]

        if edge_type_p == 'directed':
            df_match = df_G[
                (df_G['edge_type'] == 'directed') &
                (df_G['edge_label'] == edge_label_p) &
                (df_G['src_label'] == src_label_p) &
                (df_G['dst_label'] == dst_label_p)
            ][['src_id', 'dst_id']].copy()
            df_match.columns = [ui, uj]
        else:  # undirected
            df_match = df_G[
                (df_G['edge_type'] == 'undirected') &
                (df_G['edge_label'] == edge_label_p) &
                (
                    ((df_G['src_label'] == src_label_p) & (df_G['dst_label'] == dst_label_p)) |
                    ((df_G['src_label'] == dst_label_p) & (df_G['dst_label'] == src_label_p))
                )
            ][['src_id', 'dst_id']].copy()

            df_match_rev = df_match.copy()
            df_match_rev.columns = [uj, ui]
            df_match.columns = [ui, uj]

            df_match = pd.concat([df_match, df_match_rev], ignore_index=True)

        df_match.drop_duplicates(inplace=True)

        R_list.append(df_match)

    if not R_list:
        result_df = pd.DataFrame()
    else:
        result_df = R_list[0]
        for next_df in R_list[1:]:
            result_df = result_df.merge(next_df, how='inner')

    if result_df.empty:
        result_df = pd.DataFrame(columns=variable_names)
    else:
        for var in variable_names:
            if var not in result_df.columns:
                result_df[var] = np.nan

        result_df = result_df[variable_names]

        result_df.dropna(inplace=True)

        result_df = result_df.astype(int)

    result_df.sort_values(by=variable_names, inplace=True)

    output_file = os.path.join('out', 'pandas_q2.csv')
    result_df.to_csv(output_file, index=False, header=False)

    return output_file


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
