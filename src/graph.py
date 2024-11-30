'''
    Author: University of Illinois at Urbana Champaign
    DateTime: 2023-11-05 22:56:29
    FilePath: src/graph.py
    Description:
'''
import pandas as pd
import numpy as np

class Graph(object):
    def __init__(self, name: str, file_path: str):
        self._name = name
        self._path = file_path
        self._is_undirected = True
        self._vertex_count = -1
        self._edge_count = -1
        self.df = None

        if not file_path:
            self._vertex_count = 0
            self._edge_count = 0
            self._is_undirected = True
            self.df = pd.DataFrame(columns=['src_id', 'dst_id', 'src_label', 'dst_label', 'edge_label'])
            return

        df = pd.read_csv(
            file_path,
            sep=' ',
            header=None,
            names=['src_id', 'dst_id', 'src_label', 'dst_label', 'edge_label'],
            dtype={
                'src_id': np.int32,
                'dst_id': np.int32,
                'src_label': 'category',
                'dst_label': 'category',
                'edge_label': 'category'
            }
        )

        # Precompute edge attributes
        df['min_id'] = df[['src_id', 'dst_id']].min(axis=1)
        df['max_id'] = df[['src_id', 'dst_id']].max(axis=1)
        df['edge_key'] = list(zip(df['min_id'], df['max_id'], df['edge_label']))

        # Compute counts and edge_type
        edge_counts = df.groupby('edge_key').size()
        df['count'] = df['edge_key'].map(edge_counts)
        df['edge_type'] = np.where(df['count'] == 2, 1, 0)  # 1 for undirected, 0 for directed

        self.df = df

        vertices = pd.concat([df['src_id'], df['dst_id']]).unique()
        self._vertex_count = len(vertices)

        self._edge_count = len(edge_counts)
        self._is_undirected = all(edge_counts == 2)

    def print_statistics(self) -> None:
        density = self._edge_count / self._vertex_count
        if self._is_undirected:
            density *= 2
        print(f'{self._name}: {self._path}')
        print(f'vertex count {self._vertex_count}'
            + f' edge count {self._edge_count}'
            + f' density {density:.2f}')
