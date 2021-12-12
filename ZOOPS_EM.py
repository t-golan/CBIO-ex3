import argparse
from Bio import SeqIO
import pandas as pd
import numpy as np


def get_seq(fasta):
    seq_lst = []
    for f in fasta:
        seq_lst.append("^" + str(f.seq) + "$")
    return seq_lst

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


def initial_emissions(alpha, motif):
    # initalize emissions dataframe for motif, filled with alpha
    k = len(motif)
    df = pd.DataFrame(
        {"A": np.full(k, alpha),
         "C": np.full(k, alpha),
         "G": np.full(k, alpha),
         "T": np.full(k, alpha)})
    # for each state in motif update emission to 1-3*alpha
    for i in range(k):
        df[motif[i]][i] = 1 - (3*alpha)
    df['^'] = np.zeros(df.shape[0])
    df['$'] = np.zeros(df.shape[0])
    # add emissions for states not in motif
    rows = pd.DataFrame(np.zeros((4, df.shape[1])), columns=df.columns)
    df = pd.concat([rows, df], ignore_index=True)
    # 3 represents 'Bend', which emits '$' with probability 1
    df['$'][3] = 1
    # 0 represents 'Bstart', which emits '^' with probability 1
    df['^'][0] = 1
    # 1 and 2 represent B1 and B2, which emit AGCT with probability 0.25
    df.iloc[1:3, 0:4] = 0.25
    # return log
    with np.errstate(divide='ignore'):
        return df.apply(np.log)


def main():
    args = parse_args()

    # build transitions 

    # build emissions
    initial_ems = initial_emissions(args.alpha, args.seed)
    transition_mat = motif_find.transition(args.p, args.q, len(args.seed) + motif_find.EXTERNAL_STATES)
    # build emissions

    # load fasta
    seq_lst = get_seq(SeqIO.parse(open(args.fasta), 'fasta'))
    print(seq_lst)

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

