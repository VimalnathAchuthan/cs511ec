'''
    Author: University of Illinois at Urbana Champaign
    DateTime: 2023-11-05 22:56:29
    FilePath: src/graph.py
    Description:
'''
import pandas as pd

class Graph(object):
    def __init__(self, name: str, file_path: str):
        self._name = name
        self._path = file_path
        
        self._data = pd.read_csv(file_path, sep=' ', header=None)
        
        self._is_undirected = self._check_undirected()
        
        self._vertex_count = self._count_vertices()
        self._edge_count = self._count_edges()

    def _check_undirected(self) -> bool:
        edges = set(tuple(sorted(row[:2])) for row in self._data.values)
        return len(edges) < len(self._data)

    def _count_vertices(self) -> int:
        return len(set(self._data[0]).union(set(self._data[1])))

    def _count_edges(self) -> int:
        if self._is_undirected:
            return len(set(tuple(sorted(row[:2])) for row in self._data.values))
        return len(self._data)

    def print_statistics(self) -> None:
        density = self._edge_count / self._vertex_count
        if self._is_undirected:
            density *= 2
        print(f'{self._name}: {self._path}')
        print(f'vertex count {self._vertex_count}'
            + f' edge count {self._edge_count}'
            + f' density {density:.2f}')
