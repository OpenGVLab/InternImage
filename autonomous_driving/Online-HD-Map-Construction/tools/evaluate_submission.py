import sys
import os
sys.path.append(os.path.abspath('.'))    
from src.datasets.evaluation.vector_eval import VectorEvaluate
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a submission file')
    
    parser.add_argument('submission', 
        help='submission file in pickle or json format to be evaluated')

    parser.add_argument('gt', 
        help='gt annotation file')

    args = parser.parse_args()
    return args

def main(args):
    evaluator = VectorEvaluate(args.gt, n_workers=0)
    results = evaluator.evaluate(args.submission)

    print(results)

if __name__ == '__main__':
    args = parse_args()
    main(args)
