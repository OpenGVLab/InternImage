import os
import sys

sys.path.append(os.path.abspath('.'))
import argparse

from src.datasets.evaluation.vector_eval import VectorEvaluate


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
