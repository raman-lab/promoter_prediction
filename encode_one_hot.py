#!/usr/bin/env python
import argparse
import itertools
import numpy as np

# TODO: add encoding functions for just single or just pairwise nucleotide features
# possibly add structural feature encodings too


def encode_one_hot(sequence_list, length=19, pairwise=True):
    nucleotides = 'ACGT'
    positions = range(0, length)
    single_position_items = list(itertools.product(positions, nucleotides))
    pairwise_positions = list(itertools.combinations(positions, 2))
    pairwise_nucleotides = list(itertools.product(nucleotides, nucleotides))
    pairwise_items = list(itertools.product(pairwise_positions, pairwise_nucleotides))
    feature_matrix = []
    for seq in sequence_list:
        x = [1]
        for position, nucleotide in single_position_items:
            if seq[position] == nucleotide:
                x.append(1)
            else:
                x.append(0)
        if pairwise:
            for position_pair, nucleotide_pair in pairwise_items:
                if seq[position_pair[0]] == nucleotide_pair[0] and seq[position_pair[1]] == nucleotide_pair[1]:
                    x.append(1)
                else:
                    x.append(0)
        feature_matrix.append(x)
    return feature_matrix


def promoter_data_file_to_lists(promoter_data_file):
    """assumes file is 3 columns: 1. sequence 2. repression values 3. induction values"""
    sequences = []
    repression = []
    induction = []
    with open(promoter_data_file, 'r') as f:
        for line in f:
            seq, rep, ind = line.rstrip().split()
            sequences.append(seq)
            repression.append(float(rep))
            induction.append(float(ind))
    return sequences, repression, induction


def main_encode_one_hot(input_file):
    sequences, repression, induction = promoter_data_file_to_lists(input_file)
    feature_matrix = encode_one_hot(sequences)
    base_name = input_file.split('/')[-1].split('.')[0]
    np.save('{0}_feature_matrix.npy'.format(base_name), np.asarray(feature_matrix))
    np.save('{0}_repression.npy'.format(base_name), np.asarray(repression))
    np.save('{0}_induction.npy'.format(base_name), np.asarray(induction))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='encode dna sequences as one-hot single and pairwise features. '
                                                 'assumes input data file has 3 columns: '
                                                 '1. sequence, 2. repression value, 3. induction value')
    required = parser.add_argument_group('required')
    required.add_argument('-i', '--input_file', required=True)
    args = parser.parse_args()
    main_encode_one_hot(args.input_file)
