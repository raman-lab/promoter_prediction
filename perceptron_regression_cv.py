#!/usr/bin/env python
import argparse
import numpy as np
import sys
from linear_regression_cv import cross_validation_dict_and_combos_from_txt, training_sets_from_cv_dict, \
    plot_predicted_actual
from sklearn.neural_network import MLPRegressor


def perceptron_regression_cv(cross_validation_files, features):
    k = len(cross_validation_files)
    if features == 'pairwise':
        cross_validation_dict, cross_validation_combos = \
            cross_validation_dict_and_combos_from_txt(cross_validation_files, pairwise=True)
    elif features == 'single':
        cross_validation_dict, cross_validation_combos = \
            cross_validation_dict_and_combos_from_txt(cross_validation_files, pairwise=False)
    else:
        raise Exception('value passed to -f argument is not supported.')

    for cv_combo in cross_validation_combos:
        x_train, rep_train, ind_train, fi_train = training_sets_from_cv_dict(cv_combo, cross_validation_dict)
        missing_set = list(set(range(k)) - set(cv_combo))[0]
        x_validation = cross_validation_dict[missing_set]['X']
        rep_validation = cross_validation_dict[missing_set]['rep']
        ind_validation = cross_validation_dict[missing_set]['ind']
        x_validation = np.asarray(x_validation)
        fi_validation = np.log10(np.asarray(ind_validation) / np.asarray(rep_validation))
        rep_validation = np.log10(rep_validation)
        ind_validation = np.log10(ind_validation)

        # data for model to predict both repression and induction values
        rep_ind_train = np.asarray([rep_train, ind_train])
        rep_ind_validation = np.asarray([rep_validation, ind_validation])

        model_rep = MLPRegressor().fit(x_train, rep_train)
        model_ind = MLPRegressor().fit(x_train, ind_train)
        model_fi = MLPRegressor().fit(x_train, fi_train)
        model_rep_ind = MLPRegressor().fit(x_train, rep_ind_train)

        y_hat_rep = model_rep.predict(x_validation)
        y_hat_ind = model_ind.predict(x_validation)
        y_hat_fi = model_fi.predict(x_validation)
        y_hat_rep_ind = model_rep_ind.predict(x_validation)

        r_squared_rep = model_rep.score(x_validation, rep_validation)
        r_squared_ind = model_ind.score(x_validation, ind_validation)
        r_squared_fi = model_fi.score(x_validation, fi_validation)
        r_squared_rep_ind = model_rep_ind.score(x_validation, rep_ind_validation)

        sys.stdout.write('validation set {0}: rep: {1} ind: {2} both: {3} fi: {4}\n'.format(
            missing_set, r_squared_rep, r_squared_ind, r_squared_rep_ind, r_squared_fi))

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
        # plotting function does not work with both rep and fi data - should edit to plot in 3d (or some other
        # format) if handed multi dim data




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""trains a two layer perceptron regressor on input cross 
    validation data sets. models are made to predict fold induction, repression, induction and both repression and 
    indcution together""")
    parser.add_argument('-f', '--features', default='pairwise',
                        help='features to be included in model. accepted values include:'
                             'single, pairwise')
    required = parser.add_argument_group('required')
    required.add_argument('-cv', '--cross_validation_files', required=True, nargs='*',
                          help='cross validation data files')
    args = parser.parse_args()
    perceptron_regression_cv(args.cross_validation_files, args.features)
