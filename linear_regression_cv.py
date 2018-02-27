#!/usr/bin/env python
import argparse
import itertools
import numpy as np
import sys
from encode_one_hot import promoter_data_file_to_lists, encode_one_hot
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV, LinearRegression
from sklearn.neural_network import MLPRegressor
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
    fi_train = np.log10(np.asarray(ind_train) / np.asarray(rep_train))
    rep_train = np.log10(rep_train)
    ind_train = np.log10(ind_train)
    return x_train, rep_train, ind_train, fi_train


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
    plt.close()


def cross_validation_dict_and_combos_from_txt(cross_validation_files):
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
        cross_validation_dict[c]['X'] = encode_one_hot(seq_list)
        cross_validation_dict[c]['rep'] = rep_list
        cross_validation_dict[c]['ind'] = ind_list
    cross_validation_sets = itertools.combinations(range(k), k - 1)
    return cross_validation_dict, cross_validation_sets


def linear_regression_cv(cross_validation_files, lasso, ridge, least_squares, neural_network, alpha):
    cross_validation_dict, cross_validation_combos = cross_validation_dict_and_combos_from_txt(cross_validation_files)

    if not alpha:
        x_train, rep_train, ind_train, fi_train = training_sets_from_cv_dict(range(k), cross_validation_dict)
        if lasso:
            model_rep = LassoCV(cv=5).fit(x_train, rep_train)
            model_ind = LassoCV(cv=5).fit(x_train, ind_train)
            model_fi = LassoCV(cv=5).fit(x_train, fi_train)
            alpha_rep = model_rep.alpha_
            alpha_ind = model_ind.alpha_
            alpha_fi = model_fi.alpha_
        elif ridge:
            model_rep = RidgeCV(alphas=np.logspace(-4, 3), cv=5).fit(x_train, rep_train)
            model_ind = RidgeCV(alphas=np.logspace(-4, 3), cv=5).fit(x_train, ind_train)
            model_fi = RidgeCV(alphas=np.logspace(-4, 3), cv=5).fit(x_train, fi_train)
            alpha_rep = model_rep.alpha_
            alpha_ind = model_ind.alpha_
            alpha_fi = model_fi.alpha_
        elif least_squares or neural_network:
            alpha_rep = 0
            alpha_ind = 0
            alpha_fi = 0
        else:
            raise Exception('regression method not specified')
        sys.stdout.write('alpha: rep: {0} ind: {1} fi: {2}\n'.format(alpha_rep, alpha_ind, alpha_fi))
    else:
        alpha_rep = alpha
        alpha_ind = alpha
        alpha_fi = alpha

    for cv_set in cross_validation_combos:
        x_train, rep_train, ind_train, fi_train = training_sets_from_cv_dict(cv_set, cross_validation_dict)
        missing_set = list(set(range(k)) - set(cv_set))[0]
        x_validation = cross_validation_dict[missing_set]['X']
        rep_validation = cross_validation_dict[missing_set]['rep']
        ind_validation = cross_validation_dict[missing_set]['ind']
        x_validation = np.asarray(x_validation)
        fi_validation = np.log10(np.asarray(ind_validation) / np.asarray(rep_validation))
        rep_validation = np.log10(rep_validation)
        ind_validation = np.log10(ind_validation)

        if lasso:
            model_rep = Lasso(alpha=alpha_rep).fit(x_train, rep_train)
            model_ind = Lasso(alpha=alpha_ind).fit(x_train, ind_train)
            model_fi = Lasso(alpha=alpha_fi).fit(x_train, fi_train)
        elif ridge:
            model_rep = Ridge(alpha=alpha_rep).fit(x_train, rep_train)
            model_ind = Ridge(alpha=alpha_ind).fit(x_train, ind_train)
            model_fi = Ridge(alpha=alpha_fi).fit(x_train, fi_train)
        elif least_squares:
            model_rep = LinearRegression().fit(x_train, rep_train)
            model_ind = LinearRegression().fit(x_train, ind_train)
            model_fi = LinearRegression().fit(x_train, fi_train)
        elif neural_network:
            model_rep = MLPRegressor().fit(x_train, rep_train)
            model_ind = MLPRegressor().fit(x_train, ind_train)
            model_fi = MLPRegressor().fit(x_train, fi_train)
        else:
            raise Exception('regression method not specified')

        # beta_hat_rep = model_rep.coef_
        # beta_hat_ind = model_ind.coef_
        # beta_hat_fi = model_fi.coef_
        # y_hat_rep = np.dot(x_validation, beta_hat_rep)
        # y_hat_ind = np.dot(x_validation, beta_hat_ind)
        # y_hat_fi = np.dot(x_validation, beta_hat_fi)
        # r_squared_rep = r2_score(rep_validation, y_hat_rep)
        # r_squared_ind = r2_score(ind_validation, y_hat_ind)
        # r_squared_fi = r2_score(fi_validation, y_hat_fi)
        y_hat_rep = model_rep.predict(x_validation)
        y_hat_ind = model_ind.predict(x_validation)
        y_hat_fi = model_fi.predict(x_validation)
        r_squared_rep = model_rep.score(x_validation, rep_validation)
        r_squared_ind = model_ind.score(x_validation, ind_validation)
        r_squared_fi = model_fi.score(x_validation, fi_validation)
        sys.stdout.write('validation set {0}: rep: {1} ind: {2} fi: {3}\n'.format(
            missing_set, r_squared_rep, r_squared_ind, r_squared_fi))
        # t1 = np.corrcoef(rep_validation, y_hat_rep)[0, 1]
        # t2 = np.corrcoef(ind_validation, y_hat_ind)[0, 1]
        base_name = cross_validation_files[0].split('/')[-1].split('_cv_')[0]
        plot_predicted_actual(
            rep_validation, y_hat_rep, name='{0}_repression_valid_{1}'.format(base_name, missing_set)
        )
        plot_predicted_actual(
            ind_validation, y_hat_ind, name='{0}_induction_valid_{1}'.format(base_name, missing_set)
        )
        plot_predicted_actual(
            fi_validation, y_hat_fi, name='{0}_fi_valid_{1}'.format(base_name, missing_set)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='runs linear regression using either Lasso or Ridge on input cross validation data sets. single '
                    'and pairwise nucleotides are used as features. regression models are made to predict repression '
                    'and induction values, assumed to be columns 2 and 3, respectively. if alpha is not provided, '
                    'it will be selected through error minimization with cross validation')
    parser.add_argument('-a', '--alpha', type=float, help='alpha parameter used in regularization by Lasso and Ridge')
    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument('-l', '--lasso', action='store_true', help='use lasso regression')
    method_group.add_argument('-r', '--ridge', action='store_true', help='use ridge regression')
    method_group.add_argument('-s', '--least_squares', action='store_true', help='use least squares regression')
    method_group.add_argument('-n', '--neural_network', action='store_true',
                              help='use perceptron regressor with 2 layers')
    required = parser.add_argument_group('required')
    required.add_argument('-cv', '--cross_validation_files', required=True, nargs='*',
                          help='cross validation data files')
    args = parser.parse_args()
    linear_regression_cv(args.cross_validation_files, args.lasso, args.ridge,
                           args.least_squares, args.neural_network, args.alpha)
