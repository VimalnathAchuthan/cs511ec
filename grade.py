import argparse
import logging

from src import pandas_q1, pandas_q2, optimizer, executor

logger = logging.getLogger()

############################ define the parameters ############################
parser = argparse.ArgumentParser(description='grader for cs511 extra credit.')

parser.add_argument('--d1', type=str, default='data/1.txt',
    help='parameter for data file')
parser.add_argument('--d2', type=str, default='data/1.txt',
    help='parameter for data file')
parser.add_argument('--d3', type=str, default='data/2.txt',
    help='parameter for data file')
parser.add_argument('--d4', type=str, default='data/2.txt',
    help='parameter for data file')
parser.add_argument('--p1', type=str, default='pattern/1.txt',
    help='parameter for pattern file')
parser.add_argument('--p2', type=str, default='pattern/2.txt',
    help='parameter for pattern file')
parser.add_argument('--p3', type=str, default='pattern/3.txt',
    help='parameter for pattern file')
parser.add_argument('--p4', type=str, default='pattern/4.txt',
    help='parameter for pattern file')
parser.add_argument('--eq1', type=str, default='expect/d1-p1.txt',
    help='link,name download link and the expect file name')
parser.add_argument('--eq2', type=str, default='expect/d1-p2.txt',
    help='link,name download link and the expect file name')
parser.add_argument('--eq3', type=str, default='expect/d2-p3.txt',
    help='link,name download link and the expect file name')
parser.add_argument('--row', type=int, default=1666269,
    help='expect number of rows')
parser.add_argument('--column', type=int, default=7,
    help='expect number of columns')

args = parser.parse_args()
###############################################################################


#################################### test ####################################
score = 0

score += pandas_q1.test(args.d1, args.p1, args.eq1)
score += pandas_q2.test(args.d2, args.p2, args.eq2)
score += optimizer.test(args.d3, args.p3)
score += executor.test_match(args.d3, args.p3, args.eq3)
score += executor.test_count(args.d4, args.p4, args.row, args.column)

###################### print the total score ##################################
print(f"total score is: {score}/100.")
