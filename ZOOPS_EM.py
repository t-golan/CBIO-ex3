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

def get_qp_round(transiton_matrix):
    """
    Gets a transitions matrix and returns a df with only q, p values rounded within 4th digits
    """
    return pd.DataFrame(data=(transiton_matrix[0,1], transiton_matrix[1,4])).round(4)


def expectation_phase(emission_mat, transition_mat, seq_lst):
    # N_kx is size of emissions matrix (only for AGCT)
    N_kx = np.full((transition_mat.shape[0], 4), np.NINF)
    # Nkl is size of transitions matrix
    N_kl = np.full((transition_mat.shape[0], transition_mat.shape[0]), np.NINF)

    sum_ll = 0
    for seq in seq_lst:
        post_ems, post_trans, likelihood = trans_ems_post(transition_mat, emission_mat, seq)
        sum_ll += likelihood
        with np.errstate(divide='ignore'):
            N_kl = np.logaddexp(N_kl, post_trans)
        #iterate over ACGT (in same order as they appear in emissions df)
        for j, ch in enumerate("ACGT"):
            # get all indexes of letter in sequence
            idx = [i for i, ltr in enumerate(seq) if ltr == ch]
            if len(idx) != 0:
                # logsumexp - add the posterior of kx over all appearances in seq
                kx_in_seq = logsumexp(post_ems.take(idx, axis=1), axis=1)
                N_kx[:, j] = np.logaddexp(N_kx[:, j], kx_in_seq)
    return N_kx, N_kl, sum_ll


def maximization_phase(N_kx, N_kl):
    # calculate updated p and q and update transitions
    p = np.exp(np.logaddexp(N_kl[1][4], N_kl[2][3]) - logsumexp(N_kl[[1, 2], :]))
    q = np.exp(N_kl[0][1] - logsumexp(N_kl[0, :]))
    trans = motif_find.transition(p, q, N_kl.shape[0])

    ### old code - problematic
    # trans = N_kl - logsumexp(N_kl, axis=0)

    with np.errstate(all='ignore'):
        ### old code - problematic
        #ems = N_kx - logsumexp(N_kx, axis=0)
        ems = N_kx - np.array([logsumexp(N_kx, axis=1)]).T
    ems[np.isnan(ems)] = np.NINF
    # make emissions into dataframe and add ^ and $ values
    ems = pd.DataFrame(ems, columns=["A", "C", "G", "T"])
    ems['^'] = np.full(ems.shape[0], np.NINF)
    ems['$'] = np.full(ems.shape[0], np.NINF)
    ems['$'][3] = 0
    ems['^'][0] = 0
    ems.iloc[1:3, 0:4] = np.log(0.25)
    return ems, trans



def Baum_Welch_iteration(transition_mat, emission_mat, seq_lst):
    N_kx, N_kl, sum_ll = expectation_phase(emission_mat, transition_mat, seq_lst)
    emission_mat, transition_mat = maximization_phase(N_kx, N_kl)
    return sum_ll, emission_mat, transition_mat


def trans_ems_post(trans, ems, seq):
    f = Forward(seq, ems, trans)
    b = Backward(seq, ems, trans)
    likelihood = b.get_backward_prob()
    # emissions posterior probability
    ems_post = Posterior(f, b, seq).get_matrix()

    trans_post = np.full(trans.shape, np.NINF)
    for i in range(1, len(seq)):

        trans_post = np.logaddexp(trans_post,
                                  (f.get_matrix()[:, i - 1].reshape(-1, 1) + trans + ems[seq[i]].to_numpy() +
                                  b.get_matrix()[:, i].reshape(-1, 1).T - likelihood))
    return ems_post, trans_post, likelihood



def main():
    args = parse_args()

    # build transitions
    trans = motif_find.transition(args.p, args.q, len(
        args.seed) + motif_find.EXTERNAL_STATES)

    # build emissions
    ems = initial_emissions(args.alpha, args.seed)

    # load fasta
    seq_lst = get_seq(SeqIO.parse(open(args.fasta), 'fasta'))

    prev_iter = np.NINF
    # run Baum-Welch
    with open("ll_history.txt", 'w') as ll_history:
        cur_iter, ems, trans = Baum_Welch_iteration(trans, ems, seq_lst)
        ll_history.write(str(cur_iter) + '\n')
        while cur_iter - prev_iter >= args.convergenceThr:
            prev_iter = cur_iter
            cur_iter, ems, trans = Baum_Welch_iteration(trans, ems, seq_lst)
            ll_history.write(str(cur_iter) + '\n')


    # dump results

    #### old code - problematic format
    #ems.round(2)
    #qp = get_qp_round(trans)
    #motif_profile.write(pd.DataFrame.to_string(ems))
    #motif_profile.write(pd.DataFrame.to_string(qp))

    relevant_emissions = np.exp(ems.iloc[4:, :4].T).to_numpy()
    np.savetxt("motif_profile.txt", relevant_emissions, fmt='%.2f', delimiter="\t")

    with open("motif_profile.txt", "a") as motif_profile:
        p = trans[1][4]
        q = trans[0][1]
        motif_profile.write(str(round(np.exp(p), 4)) + "\n" + str(round(np.exp(q), 4)))


    with open("motif_positions.txt", 'w') as motif_positions:
        for seq in seq_lst:
            idx = Viterbi(seq, ems, trans).get_motif_index()
            motif_positions.write(str(idx) + '\n')


if __name__ == "__main__":
    main()

