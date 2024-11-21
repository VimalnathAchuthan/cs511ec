'''
    Author: University of Illinois at Urbana Champaign
    DateTime: 2023-11-05 22:24:31
    FilePath: src/optimizer.py
    Description:
'''


from src.graph import Graph
from src.report import get_file, report
import pandas as pd
import numpy as np
import itertools
import os


class Optimizer(object):
    def __init__(self, data: str, pattern: str):
        self._data_file = data
        self._pattern_file = pattern
        self._plan = []
        # TODO: begin of your codes
        '''
        self._plan explanation:
            1. represent the computation cost estimation of step by step
            2. its value should be correctly set in self.run() method
            3. initialized as a dummy value
            4. each item of self._plan should be length 2
            5. the first is of type str
            6. the second is of type int/float
        requirement:
            1. [optional] can define and initialize new fields here
            2. please put expensive computation inside self.run()
        '''
        # TODO: end of your codes

    @report
    def run(self):
        self._data = Graph('data', self._data_file)
        self._pattern = Graph('pattern', self._pattern_file)

        sample_fraction = 0.01
        df_sampled = self._data.df.sample(frac=sample_fraction, random_state=42)
        df_sampled = df_sampled.reset_index(drop=True)

        sample_file_path = os.path.join('out', 'sample_graph.txt')
        os.makedirs('out', exist_ok=True)
        df_sampled.to_csv(sample_file_path, sep=' ', index=False, header=False)

        self._sample = Graph('sample', sample_file_path)

        df_pattern = self._pattern.df
        pattern_vertices = pd.unique(df_pattern[['src_id', 'dst_id']].values.ravel('K'))
        pattern_vertices.sort()
        variable_names = ['u{}'.format(i+1) for i in range(len(pattern_vertices))]
        dict_pv = dict(zip(pattern_vertices, variable_names))
        dict_pv_rev = {v: k for k, v in dict_pv.items()}

        relations = []

        df_P = df_pattern.copy()
        df_P['min_id'] = df_P[['src_id', 'dst_id']].min(axis=1)
        df_P['max_id'] = df_P[['src_id', 'dst_id']].max(axis=1)
        df_P['edge_key'] = df_P['min_id'].astype(str) + '_' + df_P['max_id'].astype(str) + '_' + df_P['edge_label'].astype(str)
        df_P_counts = df_P.groupby('edge_key').size().reset_index(name='count')
        df_P = df_P.merge(df_P_counts, on='edge_key', how='left')
        df_P['edge_type'] = df_P['count'].apply(lambda x: 'undirected' if x == 2 else 'directed')

        df_pattern_edges = df_P[['src_id', 'dst_id', 'src_label', 'dst_label', 'edge_label', 'edge_type']].drop_duplicates()

        df_sample = self._sample.df.copy()
        df_sample['min_id'] = df_sample[['src_id', 'dst_id']].min(axis=1)
        df_sample['max_id'] = df_sample[['src_id', 'dst_id']].max(axis=1)
        df_sample['edge_key'] = df_sample['min_id'].astype(str) + '_' + df_sample['max_id'].astype(str) + '_' + df_sample['edge_label'].astype(str)
        df_sample_counts = df_sample.groupby('edge_key').size().reset_index(name='count')
        df_sample = df_sample.merge(df_sample_counts, on='edge_key', how='left')
        df_sample['edge_type'] = df_sample['count'].apply(lambda x: 'undirected' if x == 2 else 'directed')

        for idx, edge_p in df_pattern_edges.iterrows():
            src_id_p = edge_p['src_id']
            dst_id_p = edge_p['dst_id']
            src_label_p = edge_p['src_label']
            dst_label_p = edge_p['dst_label']
            edge_label_p = edge_p['edge_label']
            edge_type_p = edge_p['edge_type']

            ui = dict_pv[src_id_p]
            uj = dict_pv[dst_id_p]

            if edge_type_p == 'directed':
                df_match = df_sample[
                    (df_sample['edge_type'] == 'directed') &
                    (df_sample['edge_label'] == edge_label_p) &
                    (df_sample['src_label'] == src_label_p) &
                    (df_sample['dst_label'] == dst_label_p)
                ][['src_id', 'dst_id']].copy()
                df_match.columns = [ui, uj]
            else:
                df_match = df_sample[
                    (df_sample['edge_type'] == 'undirected') &
                    (df_sample['edge_label'] == edge_label_p) &
                    (
                        ((df_sample['src_label'] == src_label_p) & (df_sample['dst_label'] == dst_label_p)) |
                        ((df_sample['src_label'] == dst_label_p) & (df_sample['dst_label'] == src_label_p))
                    )
                ][['src_id', 'dst_id']].copy()
                df_match_rev = df_match.copy()
                df_match_rev.columns = [uj, ui]
                df_match.columns = [ui, uj]
                df_match = pd.concat([df_match, df_match_rev], ignore_index=True)

            df_match.drop_duplicates(inplace=True)
            relation_size = len(df_match)

            relations.append({'name': f'{ui},{uj}', 'size': relation_size, 'variables': set([ui, uj])})

        relations.sort(key=lambda x: x['size'])

        plan = []
        used_variables = set()
        remaining_relations = relations.copy()

        while remaining_relations:
            found = False
            for idx, rel in enumerate(remaining_relations):
                if not used_variables or used_variables & rel['variables']:
                    plan.append([rel['name'], rel['size']])
                    used_variables |= rel['variables']
                    remaining_relations.pop(idx)
                    found = True
                    break
            if not found:
                rel = remaining_relations.pop(0)
                plan.append([rel['name'], rel['size']])
                used_variables |= rel['variables']

        self._plan = plan

    def check_plan(self) -> bool:
        # do not modify this method
        self._data.print_statistics()
        self._pattern.print_statistics()
        self._sample.print_statistics()
        state = set()
        step = set()
        for item, _ in self._plan:
            step = set(map(lambda x:x.strip(), item.split(',')))
            if state and (not (state & step)):
                print(f'previous state: {",".join(sorted(state))}')
                print(f'current step  : {",".join(step)}')
                return False
            state |= step
        if self._pattern._vertex_count != len(state):
            print(f'actual state size: {len(state)}')
            print(f'expect state size: {self._pattern._vertex_count}')
            return False
        return True


def test(data, pattern) -> int:
    # import the logger to output message
    import logging
    logger = logging.getLogger()
    data = get_file('data', data)
    pattern = get_file('pattern', pattern)

    # run the test
    print("**************begin Optimizer test**************")
    opt = Optimizer(data, pattern)
    opt.run()
    try:
        assert(opt.check_plan())
        print('*******************pass*******************')
        return 10
    except Exception as e:
        logger.error("Exception Occurred:" + str(e))
        print('*******************fail*******************')
        print('optimizer.check_plan() fails.')
        return 0


if __name__ == "__main__":
    data = 'data/2.txt'
    pattern = 'pattern/3.txt'
    test(data, pattern)
