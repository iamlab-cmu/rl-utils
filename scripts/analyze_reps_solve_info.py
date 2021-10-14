import argparse
import sys

from rl_utils.analysis import reps_solve_info_analysis

def parse_arguments(input_args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        required=True,
        help="Specifies the path to a REPS solve info dictionary.",
    )
    args = parser.parse_args(input_args)
    return args

def analyze_reps_solve_info(input_args):

    args = parse_arguments(input_args)
    reps_solve_info_analysis(args.input_path)

if __name__ == "__main__":
    analyze_reps_solve_info(sys.argv[1:])
