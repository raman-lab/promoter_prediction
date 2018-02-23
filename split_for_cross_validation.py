#!/usr/bin/env python
import argparse
import random


def split_for_cross_validation(input_file, k, p):
    test_set = []
    training_set = []
    with open(input_file, 'r') as f:
        data_lines = f.readlines()
    chunk_size = int(1 / p)
    chunked_data = [data_lines[x:x + chunk_size] for x in range(0, len(data_lines), chunk_size)]
    for data_chunk in chunked_data:
        test_item = random.choice(data_chunk)
        test_set.append(test_item)
        data_chunk.remove(test_item)
        training_set.extend(data_chunk)

    base_name = input_file.split('/')[-1].split('.')[0]

    with open('{0}_test.txt'.format(base_name), 'w') as o:
        o.writelines(test_set)

    cross_validation_dict = {key: [] for key in range(1, k + 1)}
    chunked_training_data = [training_set[x:x + k] for x in range(0, len(training_set), k)]
    if len(chunked_training_data[-1]) < k:
        del chunked_training_data[-1]

    for key in cross_validation_dict.keys():
        for t, training_chunk in enumerate(chunked_training_data):
            training_item = random.choice(training_chunk)
            cross_validation_dict[key].append(training_item)
            chunked_training_data[t].remove(training_item)

    for key, data_list in cross_validation_dict.items():
        with open('{0}_cv_{1}.txt'.format(base_name, key), 'w') as o:
            o.writelines(data_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""split input data for k-fold cross validation. data is assumed to be
     3 columns: sequence, repression, induction. rows will be sorted on ratio of induction / repression. 
     p percent of data is removed for a test set. the remaining 1 - p percent of data 
     is then used to make k cross validation sets""")
    parser.add_argument('-k', type=int, default=5, help='integer number of cross validation sets to make')
    parser.add_argument('-p', type=float, default=0.2, help='fraction of data in (0,1) to keep as test set')
    required = parser.add_argument_group('required')
    required.add_argument('-i', '--input_file', required=True,
                          help='input data file with rows of data. columns separated by white space')
    args = parser.parse_args()
    split_for_cross_validation(args.input_file, args.k, args.p)
