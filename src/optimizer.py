'''
    Author: University of Illinois at Urbana Champaign
    DateTime: 2023-11-05 22:24:31
    FilePath: src/optimizer.py
    Description:
'''


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
        self._sample = Graph('sample', '')
        # TODO: begin of your codes
        '''
        requirement:
            1. the plan can be represented in any ways
            2. you can add new fields and methods, and use them here
            3. this method can return anything, no requirements
            4. please set self._plan correctly
        '''
        # TODO: end of your codes

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
