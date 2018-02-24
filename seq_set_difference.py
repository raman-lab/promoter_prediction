#!/usr/bin/env python
import argparse
import sys


def seq_set_difference(input_file_1, input_file_2):
    with open(input_file_1, 'r') as f:
        seq_set_1 = set(f.readlines())
    with open(input_file_2, 'r') as f:
        seq_set_2 = set(f.readlines())
    diff = seq_set_1 - seq_set_2
    sys.stdout.writelines(diff)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='given two input files of sequences, this script outputs '
                                                 'set 1 \ set 2')
    required = parser.add_argument_group('required')
    required.add_argument('-i1', '--input_file_1', required=True)
    required.add_argument('-i2', '--input_file_2', required=True)
    args = parser.parse_args()
    seq_set_difference(args.input_file_1, args.input_file_2)
