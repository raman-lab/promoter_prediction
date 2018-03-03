#!/usr/bin/env python
import argparse
import itertools
import numpy as np
import sys
from encode_one_hot import promoter_data_file_to_lists, encode_one_hot
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV, LinearRegression


def data_arrays_from_cv_dict(key_list, cross_validation_dict):
    x = []
    rep = []
    ind = []
    for key in key_list:
        x.extend(cross_validation_dict[key]['X'])
        rep.extend(cross_validation_dict[key]['rep'])
        ind.extend(cross_validation_dict[key]['ind'])

    x = np.asarray(x)
    fi_train = np.log10(np.asarray(ind) / np.asarray(rep))
    rep = np.log10(rep)
    ind = np.log10(ind)
    return x, rep, ind, fi_train


def plot_predicted_actual(actual, predicted, name, plot_text):
    almost_gray = '#808080'
    almost_black = '#262626'
    max_y = max(actual + predicted)
    min_y = min(actual + predicted)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(actual, predicted, c=almost_gray, marker='o', alpha=1, edgecolor=almost_black, linewidth=0.15)

    ax1.plot([min_y, max_y], [min_y, max_y], color=almost_black, linestyle='solid', linewidth=1)
    plt.xlabel('Actual', fontsize=15)
    plt.ylabel('Predicted', fontsize=15)
    plt.title(name, fontsize=15)
    if plot_text:
        plt.text(min_y, max_y, r'$r^2 = {0}$'.format(plot_text), verticalalignment='top',
                 horizontalalignment='left')
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

    plt.savefig('{0}.png'.format(name), dpi=450)
    plt.close()


def cross_validation_dict_and_combos_from_txt(cross_validation_files, pairwise=True):
    k = len(cross_validation_files)
    cross_validation_dict = {
        key: {
            'seq': [],
            'rep': [],
            'ind': [],
            'X': [],
        } for key in range(k)}
    for c, cv_file in enumerate(cross_validation_files):
        seq_list, rep_list, ind_list = promoter_data_file_to_lists(cv_file)
        cross_validation_dict[c]['seq'] = seq_list
        cross_validation_dict[c]['X'] = encode_one_hot(seq_list, pairwise=pairwise)
        cross_validation_dict[c]['rep'] = rep_list
        cross_validation_dict[c]['ind'] = ind_list
    cross_validation_sets = itertools.combinations(range(k), k - 1)
    return cross_validation_dict, cross_validation_sets


def linear_regression_cv(cross_validation_files, model, alpha):
    k = len(cross_validation_files)
    cross_validation_dict, cross_validation_combos = cross_validation_dict_and_combos_from_txt(cross_validation_files)

    r_squared_dict = {'rep': [], 'ind': [], 'fi': [], 'both': []}
    validation_dict = {'rep': [], 'ind': [], 'fi': [], 'both': {'rep': [], 'ind': []}}
    prediction_dict = {'rep': [], 'ind': [], 'fi': [], 'both': {'rep': [], 'ind': []}}

    for cv_set in cross_validation_combos:
        x_train, rep_train, ind_train, fi_train = data_arrays_from_cv_dict(cv_set, cross_validation_dict)
        missing_set = list(set(range(k)) - set(cv_set))[0]
        x_valid, rep_valid, ind_valid, fi_valid = data_arrays_from_cv_dict(missing_set, cross_validation_dict)

        training_dict = {
            'rep': rep_train,
            'ind': ind_train,
            'fi': fi_train,
            'both': np.column_stack((rep_train, ind_train))
        }

        valid_dict = {
            'rep': rep_valid,
            'ind': ind_valid,
            'fi': fi_valid,
            'both': np.column_stack((rep_valid, ind_valid))
        }

        sys.stdout.write('validation set {0}:\nr squared: '.format(missing_set))

        for variable in ['rep', 'ind', 'fi', 'both']:
            if model == 'lasso':
                regressor = Lasso(alpha=alpha)
            elif model == 'ridge':
                regressor = Ridge(alpha=alpha)
            elif model == 'linreg':
                regressor = LinearRegression()
            else:
                raise Exception('regression method not specified')
            regressor.fit(x_train, training_dict[variable])
            y_hat = regressor.predict(x_valid)
            r_squared = regressor.score(x_valid, valid_dict[variable])

            r_squared_dict[variable].append(r_squared)
            if variable == 'both':
                validation_dict[variable]['rep'].extend(valid_dict[variable][:, 0])
                validation_dict[variable]['ind'].extend(valid_dict[variable][:, 1])
                prediction_dict[variable]['rep'].extend(y_hat[:, 0])
                prediction_dict[variable]['ind'].extend(y_hat[:, 1])
            else:
                validation_dict[variable].extend(valid_dict[variable])
                prediction_dict[variable].extend(y_hat)

            sys.stdout.write('{0}: {1}\t'.format(variable, r_squared))
        sys.stdout.write('\n')

    base_name = cross_validation_files[0].split('/')[-1].split('_cv_')[0]
    sys.stdout.write('avg r squared: ')
    for variable in ['rep', 'ind', 'fi', 'both']:
        r_squared_avg = round(np.average(r_squared_dict[variable]), 2)
        r_squared_std = round(np.std(r_squared_dict[variable]), 2)
        sys.stdout.write('{0}: {1} +/- {2}\t'.format(variable, r_squared_avg, r_squared_std))
        if variable == 'both':
            plot_predicted_actual(
                validation_dict[variable]['rep'], prediction_dict[variable]['rep'],
                name='{0}_{1}_{2}-fold_{3}_rep'.format(base_name, model, k, variable),
                plot_text='{0} \pm {1}'.format(r_squared_avg, r_squared_std)
            )
            plot_predicted_actual(
                validation_dict[variable]['ind'], prediction_dict[variable]['ind'],
                name='{0}_{1}_{2}-fold_{3}_ind'.format(base_name, model, k, variable),
                plot_text='{0} \pm {1}'.format(r_squared_avg, r_squared_std)
            )
        else:
            plot_predicted_actual(
                validation_dict[variable], prediction_dict[variable],
                name='{0}_{1}_{2}-fold_{3}'.format(base_name, model, k, variable),
                plot_text='{0} \pm {1}'.format(r_squared_avg, r_squared_std)
            )
    sys.stdout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='runs linear regression using either Lasso or Ridge on input cross validation data sets. single '
                    'and pairwise nucleotides are used as features. regression models are made to predict repression '
                    'and induction values, assumed to be columns 2 and 3, respectively.')
    parser.add_argument('-a', '--alpha', type=float, default=1.0,
                        help='alpha parameter used in regularization by Lasso and Ridge')
    parser.add_argument('-m', '--model', choices=['linreg', 'lasso', 'ridge'], default='linreg',
                        help='type of linear regression model to use')

    required = parser.add_argument_group('required')
    required.add_argument('-cv', '--cross_validation_files', required=True, nargs='*',
                          help='cross validation data files')
    args = parser.parse_args()
    linear_regression_cv(args.cross_validation_files, args.model, args.alpha)
