#!/usr/bin/env python
"""script that takes input text files with one sequence per line and outputs fasta formatted
file with those sequences to stdout"""
import sys


def txt_to_fas(input_txt_files):
    counter = 0
    for txt_file in input_txt_files:
        with open(txt_file, 'r') as f:
            for line in f:
                sys.stdout.write('>seq_{0}\n'.format(counter))
                sys.stdout.write(line)
                counter += 1


if __name__ == '__main__':
    input_files = sys.argv[1:]
    txt_to_fas(input_files)
