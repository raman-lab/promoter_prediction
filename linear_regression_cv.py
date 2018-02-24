#!/usr/bin/env python
import argparse
import itertools
import numpy as np
import sys
from encode_one_hot import promoter_data_file_to_lists, encode_one_hot
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
from sklearn.metrics import r2_score


def training_sets_from_cv_dict(key_list, cross_validation_dict):
    x_train = []
    rep_train = []
    ind_train = []
    for key in key_list:
        x_train.extend(cross_validation_dict[key]['X'])
        rep_train.extend(cross_validation_dict[key]['rep'])
        ind_train.extend(cross_validation_dict[key]['ind'])

    x_train = np.asarray(x_train)
    rep_train = np.log10(rep_train)
    ind_train = np.log10(ind_train)
    return x_train, rep_train, ind_train


def plot_predicted_actual(y_actual, y_predicted, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    max_y = max(np.append(y_actual, y_predicted))
    min_y = min(np.append(y_actual, y_predicted))
    r_squared = r2_score(y_actual, y_predicted)
    plt.plot(y_actual, y_predicted, '.')
    plt.plot([min_y, max_y], [min_y, max_y], 'k')
    plt.xlabel('Actual Y')
    plt.ylabel('Predicted Y')
    plt.text(min_y, max_y, 'rr = %0.2f' % r_squared, verticalalignment='top', horizontalalignment='left')
    plt.savefig(name)


def main_linear_regression(cross_validation_files, lasso, ridge, alpha):
    k = len(cross_validation_files)
    cross_validation_dict = {
        key: {
            'seq': [],
            'rep': [],
            'ind': [],
            'X': [],
        } for key in range(k)}
    for c, cv_file in cross_validation_files:
        seq_list, rep_list, ind_list = promoter_data_file_to_lists(cv_file)
        cross_validation_dict[k]['seq'] = seq_list
        cross_validation_dict[k]['X'] = encode_one_hot(seq_list)
        cross_validation_dict[k]['rep'] = rep_list
        cross_validation_dict[k]['ind'] = ind_list
    cross_validation_sets = itertools.combinations(range(k), k - 1)

    if not alpha:
        x_train, rep_train, ind_train = training_sets_from_cv_dict(range(k), cross_validation_dict)

        if lasso:
            model_rep = LassoCV(cv=5).fit(x_train, rep_train)
            model_ind = LassoCV(cv=5).fit(x_train, ind_train)
            alpha_rep = model_rep.alpha_
            alpha_ind = model_ind.alpha_
        elif ridge:
            model_rep = RidgeCV(alphas=np.logspace(-2, 1), cv=5).fit(x_train, rep_train)
            model_ind = RidgeCV(alphas=np.logspace(-2, 1), cv=5).fit(x_train, ind_train)
            alpha_rep = model_rep.alpha_
            alpha_ind = model_ind.alpha_
        else:
            raise Exception('regression method not specified')
        sys.stdout.write('alpha: rep: {0} ind: {1}\n'.format(alpha_rep, alpha_ind))
    else:
        alpha_rep = alpha
        alpha_ind = alpha

    for cv_set in cross_validation_sets:
        x_train, rep_train, ind_train = training_sets_from_cv_dict(cv_set, cross_validation_dict)

        missing_set = list(set(range(k)) - set(cv_set))[0]
        x_validation = cross_validation_dict[missing_set]['X']
        rep_validation = cross_validation_dict[missing_set]['rep']
        ind_validation = cross_validation_dict[missing_set]['ind']
        x_validation = np.asarray(x_validation)
        rep_validation = np.log10(rep_validation)
        ind_validation = np.log10(ind_validation)

        if lasso:
            model_rep = Lasso(alpha=alpha_rep).fit(x_train, rep_train)
            model_ind = Lasso(alpha=alpha_ind).fit(x_train, ind_train)
        elif ridge:
            model_rep = Ridge(alpha=alpha_rep).fit(x_train, rep_train)
            model_ind = Ridge(alpha=alpha_ind).fit(x_train, ind_train)
        else:
            raise Exception('regression method not specified')

        beta_hat_rep = model_rep.coef_
        beta_hat_ind = model_ind.coef_
        y_hat_rep = np.dot(x_validation, beta_hat_rep)
        y_hat_ind = np.dot(x_validation, beta_hat_ind)
        r_squared_rep = r2_score(rep_validation, y_hat_rep)
        r_squared_ind = r2_score(ind_validation, y_hat_ind)
        sys.stdout.write('validation set {0}: rep: {1} ind: {2}\n'.format(missing_set, r_squared_rep, r_squared_ind))

        base_name = cross_validation_files[0].split('/')[-1].split('_cv_')[0]
        plot_predicted_actual(
            rep_validation, y_hat_rep, name='{0}_repression_valid_{1}'.format(base_name, missing_set)
        )
        plot_predicted_actual(
            ind_validation, y_hat_ind, name='{0}_induction_valid_{1}'.format(base_name, missing_set)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='runs linear regression using either Lasso or Ridge on input cross validation data sets. single '
                    'and pairwise nucleotides are used as features. regression models are made to predict repression '
                    'and induction values, assumed to be columns 2 and 3, respectively. if alpha is not provided, '
                    'it will be selected through error minimization with cross validation')
    parser.add_argument('-a', '--alpha', type=float, help='alpha parameter used in regularization by Lasso and Ridge')
    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument('-l', '--lasso', action='store_true', help='use lasso for regression')
    method_group.add_argument('-r', '--ridge', action='store_true', help='use ridge for regression')
    required = parser.add_argument_group(required=True)
    required.add_argument('-cv', '--cross_validation_files', nargs='*', help='cross validation data files')
    args = parser.parse_args()
    main_linear_regression(args.cross_validation_files, args.lasso, args.ridge, args.alpha)
