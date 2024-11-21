'''
    Author: University of Illinois at Urbana Champaign
    DateTime: 2023-11-05 22:24:31
    FilePath: src/optimizer.py
    Description:
'''


import random
import os
import pandas as pd
from src.graph import Graph
from src.report import get_file, report

class Optimizer(object):
    def __init__(self, data: str, pattern: str):
        self._data_file = data
        self._pattern_file = pattern
        self._plan = [
            ['u1,u2', 100],
            ['u1,u3', 300],
            ['u3,u4', 20],
        ]

    @report
    def run(self):
        self._data = Graph('data', self._data_file)
        self._pattern = Graph('pattern', self._pattern_file)
        
        sample_path = self._create_sample_graph()
        self._sample = Graph('sample', sample_path)
        
        self._plan = self._compute_join_plan()

    def _create_sample_graph(self) -> str:
        sample_size = min(1000, len(pd.read_csv(self._data_file, sep=' ', header=None)))
        sample_data = pd.read_csv(self._data_file, sep=' ', header=None).sample(sample_size)
        
        sample_path = 'out/sample_graph.txt'
        os.makedirs('out', exist_ok=True)
        sample_data.to_csv(sample_path, sep=' ', index=False, header=False)
        
        return sample_path

    def _compute_join_plan(self) -> list:
        pattern_data = pd.read_csv(self._pattern_file, sep=' ', header=None)
        vertices = set(pattern_data[0]).union(set(pattern_data[1]))
        
        plan = []
        for i in range(len(vertices) - 1):
            plan.append([f'u{i+1},u{i+2}', random.randint(10, 500)])
        
        return plan

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
