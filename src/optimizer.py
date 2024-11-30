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
import os


class Optimizer(object):
    def __init__(self, data: str, pattern: str):
        self._data_file = data
        self._pattern_file = pattern
        self._plan = []

    @report
    def run(self):
        self._data = Graph('data', self._data_file)
        self._pattern = Graph('pattern', self._pattern_file)

        # Sampling 10% of the data graph
        sample_fraction = 0.1
        df_sampled = self._data.df.sample(frac=sample_fraction, random_state=42)

        # Save sampled graph to a file and reload it as a Graph object
        sample_file_path = os.path.join('out', 'sample_graph.txt')
        os.makedirs('out', exist_ok=True)
        df_sampled.to_csv(sample_file_path, sep=' ', index=False, header=False)
        self._sample = Graph('sample', sample_file_path)

        df_pattern = self._pattern.df
        pattern_vertices = np.sort(pd.unique(df_pattern[['src_id', 'dst_id']].values.ravel()))
        variable_names = ['u{}'.format(i+1) for i in range(len(pattern_vertices))]
        dict_pv = dict(zip(pattern_vertices, variable_names))

        relations = []

        df_P = df_pattern[['src_id', 'dst_id', 'src_label', 'dst_label', 'edge_label', 'edge_type']].drop_duplicates()
        df_sample = self._sample.df

        for _, edge_p in df_P.iterrows():
            src_id_p = edge_p['src_id']
            dst_id_p = edge_p['dst_id']
            src_label_p = edge_p['src_label']
            dst_label_p = edge_p['dst_label']
            edge_label_p = edge_p['edge_label']
            edge_type_p = edge_p['edge_type']

            ui = dict_pv[src_id_p]
            uj = dict_pv[dst_id_p]

            if edge_type_p == 0:  # Directed
                df_match = df_sample[
                    (df_sample['edge_type'] == 0) &
                    (df_sample['edge_label'] == edge_label_p) &
                    (df_sample['src_label'] == src_label_p) &
                    (df_sample['dst_label'] == dst_label_p)
                ][['src_id', 'dst_id']].rename(columns={'src_id': ui, 'dst_id': uj})
            else:  # Undirected
                df_match = df_sample[
                    (df_sample['edge_type'] == 1) &
                    (df_sample['edge_label'] == edge_label_p) &
                    (
                        ((df_sample['src_label'] == src_label_p) & (df_sample['dst_label'] == dst_label_p)) |
                        ((df_sample['src_label'] == dst_label_p) & (df_sample['dst_label'] == src_label_p))
                    )
                ][['src_id', 'dst_id']]
                df_match_rev = df_match.rename(columns={'src_id': uj, 'dst_id': ui})
                df_match = df_match.rename(columns={'src_id': ui, 'dst_id': uj})
                df_match = pd.concat([df_match, df_match_rev], ignore_index=True)

            df_match.drop_duplicates(inplace=True)
            relation_size = len(df_match)
            relations.append({'name': f'{ui},{uj}', 'size': relation_size, 'variables': {ui, uj}})

        # Sort relations by size to optimize join order
        relations.sort(key=lambda x: x['size'])

        # Generate execution plan
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
