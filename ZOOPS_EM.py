import argparse

import motif_find
from motif_find import *


def parse_args():
    """
    Parse the command line arguments.
    :return: The parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta', help='File path with list of sequences (e.g. seqs_ATTA.fasta)')
    parser.add_argument('seed', help='Guess for the motif (e.g. ATTA)')
    parser.add_argument('p', type=float, help='Initial guess for the p transition probability (e.g. 0.01)')
    parser.add_argument('q', type=float, help='Initial guess for the q transition probability (e.g. 0.9)')
    parser.add_argument('alpha', type=float, help='Softening parameter for the initial profile (e.g. 0.1)')
    parser.add_argument('convergenceThr', type=float, help='ll improvement threshold for the stopping condition'
                                                           ' (e.g. 0.1)')
    return parser.parse_args()


def main():
    args = parse_args()

    # build transitions 
    transition_mat = motif_find.transition(args.p, args.q, len(args.seed) + motif_find.EXTERNAL_STATES)
    # build emissions 

    # load fasta

    # run Baum-Welch
    prev_iter = Baum_Welch_iteration(transition_mat, emmisions_mat, seqs)
    while(True):
        cur_iter = Baum_Welch_iteration(transition_mat, emmisions_mat, seqs)
        if cur_iter - prev_iter < args.convergenceThr:
            break
        prev_iter = cur_iter

    # dump results


if __name__ == "__main__":
    main()

