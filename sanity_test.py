#!/usr/bin/python3 -u

import os
import argparse
import subprocess

"""
This is a sanity-test script for your ZOOPS_EM.py.
No need to read, edit or submit it this file!

Better make sure you pass this test before submitting.
Usage:
    python3 sanity_test.py PATH_TO_MOTIF_FIND
"""


def is_output_correct(ret, seed):
    """ validate user output against school solution """
    # make sure output files were created:
    for f in ('motif_profile.txt', 'motif_positions.txt', 'll_history.txt'):
        if not os.path.isfile(f):
            print(f'missing output file {f}')
            return False

    # validate motif_profile format:
    with open('motif_profile.txt', 'r') as f:
        lines = f.readlines()[:6]

    if len(lines) < 6 or not [len(l.split('\t')) for l in lines] == [len(seed)] * 4 + [1, 1]:
        print('motif_profile.txt: bad format')
        return False

    return True


def run_test(mf, fasta, seed, timeout):
    # run test as subprocess
    try:
        cmd = f'python3 {mf} {fasta} {seed} 0.05 0.9 0.1 0.1'
        ret = subprocess.check_output(cmd, shell=True, timeout=timeout).decode()
    except Exception as e:
        print(f'Failed to run {mf} as a subprocess! ', e)
        return False

    # validate return value and print SUCCESS/FAIL
    if is_output_correct(ret, seed):
        print('\033[32m{}\033[00m'.format('SUCCESS'))
    else:
        print('\033[31m{}\033[00m'.format('FAIL'))


def main(args):

    # make sure input ZOOPS_EM.py exists
    mf = args.motif_find_path
    if not os.path.isfile(mf):
        print(f'Invalid file: {mf}')
        return 1

    # generate and dump a trivial fasta file
    seed = 'CCGG'
    fasta = f'./seqs_{seed}.fasta'
    with open(fasta, 'w') as f:
        for i in range(10):
            f.write(f'>seq{i + 1}\n')
            f.write('A' * (i + 2) + seed + 'A' * (15 - i) + '\n')

    run_test(mf, fasta, seed, 2)

    # cleanup
    os.remove(fasta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('motif_find_path', help='Path to your motif_find.py script (e.g. ./ZOOPS_EM.py)')
    main(parser.parse_args())
