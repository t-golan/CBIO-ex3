import argparse
from Bio import SeqIO
import pandas as pd
import numpy as np
import motif_find


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

def get_qp(transiton_matrix):
    """
    Gets a transitions matrix and returns a df with only q, p values
    """
    return pd.DataFrame(data=(transiton_matrix[0,1], transiton_matrix[1,4]))


def get_likelihood_Nkx_Nkl(emission_mat, transition_mat, seq_lst):
    N_kx = np.full((transition_mat.shape[0], 4), np.NINF)
    N_kl = np.full((transition_mat.shape[0], transition_mat.shape[0]), np.NINF)
    sum_ll = 0
    for seq in seq_lst:
        post_ems, post_trans, likelihood = trans_ems_post(transition_mat, emission_mat, seq)
        sum_ll += likelihood
        with np.errstate(divide='ignore'):
            N_kl = np.log(np.exp(N_kl) + np.exp(post_trans))
        for j, ch in enumerate("ACGT"):
            idx = [i for i, ltr in enumerate(seq) if ltr == ch]
            if len(idx) != 0:
                N_kx[:, j] = np.logaddexp(N_kx[:, j], logsumexp(post_ems.take(idx, axis=1), axis=1))
    return N_kx, N_kl, sum_ll


def trans_ems_post(trans, ems, seq):
    fmat = Forward(seq, ems, trans).get_matrix()
    bmat = Backward(seq, ems, trans).get_matrix()
    likelihood = Backward(seq, ems, trans).get_backward_prob()
    # emissions posterior probability
    ems_post =  Posterior(fmat, bmat, seq).get_matrix()

    trans_post = np.full(trans.shape, np.NINF)
    for i in range(1, len(seq)):

        trans_post = np.logaddexp(trans_post,
                                  (fmat[:, i - 1].reshape(-1, 1) + trans + ems[seq[i]].to_numpy() +
                                  bmat[:, i].reshape(-1, 1).T - likelihood))
    return ems_post, trans_post, likelihood



def main():
    args = parse_args()

    # build transitions 

    # build emissions
    initial_ems = initial_emissions(args.alpha, args.seed)
    initial_trans = motif_find.transition(args.p, args.q, len(args.seed) + motif_find.EXTERNAL_STATES)
    # build emissions

    # load fasta
    seq_lst = get_seq(SeqIO.parse(open(args.fasta), 'fasta'))
    print(seq_lst)

    # run Baum-Welch
    ll_history = open("ll_history.txt", "w+")
    ems = initial_ems
    trans = initial_trans
    prev_iter = Baum_Welch_iteration(trans, initial_ems, seq_lst)
    ll_history.write(prev_iter + '\n')
    improvement = np.inf
    while improvement >= args.convergenceThr:
        cur_iter = Baum_Welch_iteration(trans, initial_ems, seq_lst)
        ll_history.write(cur_iter)
        improvement = cur_iter - prev_iter
        prev_iter = cur_iter

    # dump results

    ems.round(2)
    qp = get_qp(trans).round(4)
    motif_profile = open("motif_profile.txt", "w+")
    motif_profile.write(pd.DataFrame.to_string(ems))
    motif_profile.write(pd.DataFrame.to_string(qp))


if __name__ == "__main__":
    main()

