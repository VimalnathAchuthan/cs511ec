'''
    Author: University of Illinois at Urbana Champaign
    DateTime: 2023-11-05 22:56:29
    FilePath: src/graph.py
    Description:
'''


class Graph(object):
    def __init__(self, name: str, file_path: str):
        self._name = name
        self._path = file_path
        # TODO: begin of your codes
        '''
        you can add new fields and methods to this class
        please correctly set the following fields in this method
        '''
        self._is_undirected = True
        self._vertex_count = -1
        self._edge_count = -1
        # TODO: end of your codes

    def print_statistics(self) -> None:
        density = self._edge_count / self._vertex_count
        if self._is_undirected:
            density *= 2
        print(f'{self._name}: {self._path}')
        print(f'vertex count {self._vertex_count}'
            + f' edge count {self._edge_count}'
            + f' density {density:.2f}')
