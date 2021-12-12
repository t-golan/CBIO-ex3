import argparse
import pandas as pd
import numpy as np
from scipy.special import logsumexp


EXTERNAL_STATES = 4
BACKGROUND = 'B'
MOTIF = 'M'

state_to_index = {
    'Bstart': 0,
    'B1': 1,
    'B2': 2,
    'Bend': 3
}

class Forward:
    def __init__(self, seq, emissions, transitions):
        self.seq = seq
        self.emissions = emissions
        self.transitions = transitions
        self.rows = emissions.shape[0] # number of states
        self.cols = len(self.seq)
        self.mat = np.zeros((self.rows, self.cols))
        self.fill_mat()

    def fill_mat(self):
        # base case - start state is 1, rest are zero
        self.mat[0][0] = 1
        # take log
        with np.errstate(divide='ignore'):
            self.mat = np.log(self.mat)
        # dynamic programming portion:
        for i in range(1, self.cols):
            # before log:
            #self.mat[:, i] = np.multiply(self.emissions[self.seq[i]], np.matmul(self.transitions.T, self.mat[:, i - 1]))

            self.mat[:, i] = self.emissions[self.seq[i]] + \
                             logsumexp(self.mat[:, i - 1] + self.transitions.T, axis=1)

    def get_matrix(self):
        return self.mat

    def get_forward_prob(self):
        # return probability of end state in end of sequence
        return self.mat[3][-1]


class Backward:
    def __init__(self, seq, emissions, transitions):
        self.seq = seq
        self.emissions = emissions
        self.transitions = transitions
        self.rows = emissions.shape[0]  # number of states
        self.cols = len(self.seq)
        self.mat = np.zeros((self.rows, self.cols))
        self.fill_mat()

    def fill_mat(self):
        # base case - end state is 1, rest are zero
        self.mat[3][-1] = 1
        with np.errstate(divide='ignore'):
            self.mat = np.log(self.mat)
        # dynamic programming portion:
        for i in range(self.cols - 2, -1, -1):
            # before log: not sure this multiplication order gives right indexes
            #self.mat[:, i] = np.matmul(self.transitions,
            #                            self.mat[:, i + 1],
            #                           self.emissions[self.seq[i+1]].to_numpy())
            self.mat[:, i] = logsumexp(
                self.mat[:, i + 1] +
                self.emissions[self.seq[i+1]].to_numpy() + self.transitions, axis=1)

    def get_backward_prob(self):
        # return probability of beginning state in beginning of sequence
        return self.mat[0][0]

    def get_matrix(self):
        return self.mat


class Viterbi:
    def __init__(self, seq, emissions, transitions):
        self.seq = seq
        self.emissions = emissions
        self.transitions = transitions
        self.rows = emissions.shape[0]  # number of states
        self.cols = len(self.seq)
        self.v_mat = np.zeros((self.rows, self.cols))
        self.ptr_mat = np.zeros((self.rows, self.cols))
        self.fill_mat()

    def fill_mat(self):
        # base case - start state is 1, rest are zero
        self.v_mat[0][0] = 1
        with np.errstate(divide='ignore'):
            self.v_mat = np.log(self.v_mat)
        # dynamic programming portion
        for i in range(1, self.cols):
            self.v_mat[:, i] = self.emissions[self.seq[i]] + np.max(self.v_mat[:, i - 1] + self.transitions.T, axis=1)
            # keep matrix of argmaxs
            self.ptr_mat[:, i] = np.argmax(self.v_mat[:, i - 1] + self.transitions.T, axis=1)


    def get_viterbi_path(self):
        # create a vector with the indexes of the state corresponding to
        # each position
        path = np.zeros(self.v_mat.shape[1])
        argmax_k = np.argmax(self.v_mat[:, -1])
        path[-1] = argmax_k
        # go backwards over columns and add argmax of each column to path
        for i in range(self.v_mat.shape[1] - 1, 1, -1):
            path[i - 1] = self.ptr_mat[argmax_k, i]
            argmax_k = int(path[i - 1])
        # replace state indexes with B and M
        path = np.where(path < EXTERNAL_STATES, BACKGROUND, MOTIF)
        # make string and remove ^ and $ columns
        path_str = "".join(path)[1:-1]
        return path_str


class Posterior:
    def __init__(self, forward, backward, seq):
        self.seq = seq
        self.forward_mat = forward.get_matrix()
        self.backward_mat = backward.get_matrix()
        (self.rows, self.cols) = self.forward_mat.shape
        self.mat = np.zeros((self.rows, self.cols))
        self.seq_prob = forward.get_forward_prob()
        self.fill_mat()

    def fill_mat(self):
        self.mat = (self.forward_mat + self.backward_mat) - self.seq_prob
        #for i in range(self.rows):
        #    for j in range(self.cols):
        #        self.mat[i][j] = (self.forward_mat[i][j] + self.backward_mat[i][j])
        #self.mat -= self.seq_prob

    def get_hmm(self):
        # get max of each col
        states = np.argmax(self.mat, axis=0)
        # replace state indexes with B and M
        hmm = np.where(states < EXTERNAL_STATES, BACKGROUND, MOTIF)
        # make string and remove ^ and $ columns
        hmm_str = "".join(hmm)[1:-1]
        return hmm_str



def print_result(hmm, seq):
    # prints result strings in chunks of 50 chars
    for i in range(0, len(seq), 50):
        print(hmm[i:i + 50])
        print(seq[i:i + 50])
        print()


def tsv_to_emission_mat(tsv_path):
    """
    converts a tsv file to pandas df of score matrix
    :param tsv_path: a path to a tsv file
    :return: pandas df of score matrix
    """
    df = pd.read_csv(tsv_path, sep='\t')
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
    #return df
    ## added for log:
    with np.errstate(divide='ignore'):
        return df.apply(np.log)

def transition(p, q, k):
    """
    this function creates the transition matrix according to the transition
    graph in the exercise description.
    :param p:
    :param q:
    :param k: number of states(including B_start, B_end, B_1, B_2)
    :return:
    """
    trans = np.zeros((k, k))
    # Bstart -> B1
    trans[0,1] = q
    # Bstart -> M1
    trans[0,2] = 1-q
    # B1 -> B1
    trans[1,1] = 1-p
    # B1 -> M1
    trans[1,4] = p
    # B2 -> B2
    trans[2,2] = 1-p
    # Mend -> B2
    trans[k-1,2] = 1
    #B2 -> Bend
    trans[2,3] = p
    # Mi -> Mi+1
    for i in range(4, k-1):
        trans[i, i+1] = 1
    #return trans
    ## added for log:
    with np.errstate(divide='ignore'):
        return np.log(trans)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)',
                        required=True)
    parser.add_argument('seq',
                        help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
    parser.add_argument('initial_emission',
                        help='Path to emission table (e.g. initial_emission.tsv)')
    parser.add_argument('p', help='transition probability p (e.g. 0.01)',
                        type=float)
    parser.add_argument('q', help='transition probability q (e.g. 0.5)',
                        type=float)
    args = parser.parse_args()
    seq = '^' + args.seq + '$'
    emission_mat = tsv_to_emission_mat(args.initial_emission)
    transition_mat = transition(args.p, args.q, emission_mat.shape[0])
    if args.alg == 'viterbi':
        v = Viterbi(seq, emission_mat, transition_mat)
        print_result(v.get_viterbi_path(), args.seq)

    elif args.alg == 'forward':
        f = Forward(seq, emission_mat, transition_mat)
        print(f.get_forward_prob())

    elif args.alg == 'backward':
        b = Backward(seq, emission_mat, transition_mat)
        print(b.get_backward_prob())

    elif args.alg == 'posterior':
        f = Forward(seq, emission_mat, transition_mat)
        b = Backward(seq, emission_mat, transition_mat)
        p = Posterior(f, b, seq)
        print_result(p.get_hmm(), args.seq)




if __name__ == '__main__':
    main()
