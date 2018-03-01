#!/usr/bin/env python
import argparse
import numpy as np


def parse_txt_by_column(input_file):
    seq_list = []
    rep_list = []
    ind_list = []
    with open(input_file, 'r') as f:
        for line in f:
            seq, rep, ind = line.rstrip().split()
            seq_list.append(seq)
            rep_list.append(float(rep))
            ind_list.append(float(ind))
    return seq_list, rep_list, ind_list


def plot_histogram(bins, data_list, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    almost_gray = '#808080'
    almost_black = '#262626'

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.hist(data_list, bins, alpha=1, color=almost_gray, label='n = {0}'.format(len(data_list)))

    legend = ax1.legend(loc='best', framealpha=0.5)
    rect = legend.get_frame()
    rect.set_linewidth(0.0)
    texts = legend.texts
    for t in texts:
        t.set_color(almost_black)
    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax1.spines[spine].set_visible(False)
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    spines_to_keep = ['bottom', 'left']
    for spine in spines_to_keep:
        ax1.spines[spine].set_linewidth(0.5)
        ax1.spines[spine].set_color(almost_black)
    ax1.xaxis.label.set_color(almost_black)
    ax1.yaxis.label.set_color(almost_black)

    plt.title(name, fontsize=15, y=1.02)
    plt.xlabel('bins', fontsize=15)
    plt.ylabel('frequency', fontsize=15)
    plt.savefig('{0}.png'.format(name), dpi=450)
    plt.close()


def histogram_from_txt(input_file):
    seq_list, rep_list, ind_list = parse_txt_by_column(input_file)
    rep_list = np.asarray(rep_list)
    ind_list = np.asarray(ind_list)
    data_dict = {
        'rep': np.log10(rep_list),
        'ind': np.log10(ind_list),
        'fi': np.log10(ind_list / rep_list)
    }
    for key in data_dict.keys():
        bins = np.linspace(min(data_dict[key]), max(data_dict[key]))
        name = '{0}_{1}'.format(input_file.split('/')[-1].split('.')[0], key)
        plot_histogram(bins, data_dict[key], name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""make histogram of input data columns. 
    assumes input data file has 3 columns: 1. sequence, 2. repression value, 3. induction value""")
    required = parser.add_argument_group('required')
    required.add_argument('-i', '--input_file', required=True,
                          help='input data file with 3 columns: 1. sequence, 2. repression value, 3. induction value')
    args = parser.parse_args()
    histogram_from_txt(args.input_file)
