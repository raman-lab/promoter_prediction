#!/usr/bin/env python
import argparse
import numpy as np
import sys
from linear_regression_cv import cross_validation_dict_and_combos_from_txt, data_arrays_from_cv_dict, \
    plot_predicted_actual
from sklearn.neural_network import MLPRegressor

# TODO: need to change modeling fitting etc into a loop

def perceptron_regression_cv(cross_validation_files, features, nodes,
                             verbose, learning_rate, activation, tolerance):
    k = len(cross_validation_files)
    if features == 'pairwise':
        cross_validation_dict, cross_validation_combos = \
            cross_validation_dict_and_combos_from_txt(cross_validation_files, pairwise=True)
    elif features == 'single':
        cross_validation_dict, cross_validation_combos = \
            cross_validation_dict_and_combos_from_txt(cross_validation_files, pairwise=False)
    else:
        raise Exception('value passed to -f argument is not supported.')

    r_squared_dict = {'rep': [], 'ind': [], 'fi': [], 'both': []}
    validation_dict = {'rep': [], 'ind': [], 'fi': [], 'both': {'rep': [], 'ind': []}}
    prediction_dict = {'rep': [], 'ind': [], 'fi': [], 'both': {'rep': [], 'ind': []}}

    for cv_combo in cross_validation_combos:
        x_train, rep_train, ind_train, fi_train = data_arrays_from_cv_dict(cv_combo, cross_validation_dict)
        missing_set = list(set(range(k)) - set(cv_combo))
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
        sys.stdout.write('validation set {0}:\n'.format(missing_set))

        for variable in ['rep', 'ind', 'fi', 'both']:
            sys.stdout.write('training {0}:\n'.format(variable))
            regressor = MLPRegressor(
                hidden_layer_sizes=(100, 20),
                verbose=verbose,
                learning_rate_init=learning_rate,
                activation=activation,
                tol=tolerance,
            ).fit(x_train, training_dict[variable])
            y_hat = regressor.predict(x_valid)
            r_squared = regressor.score(x_valid, valid_dict[variable])
            sys.stdout.write('r squared: {0}\n'.format(r_squared))
            r_squared_dict[variable].append(r_squared)
            if variable == 'both':
                validation_dict[variable]['rep'].extend(valid_dict[variable][:, 0])
                validation_dict[variable]['ind'].extend(valid_dict[variable][:, 1])
                prediction_dict[variable]['rep'].extend(y_hat[:, 0])
                prediction_dict[variable]['ind'].extend(y_hat[:, 1])
            else:
                validation_dict[variable].extend(valid_dict[variable])
                prediction_dict[variable].extend(y_hat)

    base_name = cross_validation_files[0].split('/')[-1].split('_cv_')[0]
    sys.stdout.write('avg r squared: ')
    for variable in ['rep', 'ind', 'fi', 'both']:
        r_squared_avg = round(np.average(r_squared_dict[variable]), 2)
        r_squared_std = round(np.std(r_squared_dict[variable]), 2)
        sys.stdout.write('{0}: {1} +/- {2}\t'.format(variable, r_squared_avg, r_squared_std))
        if variable == 'both':
            plot_predicted_actual(
                validation_dict[variable]['rep'], prediction_dict[variable]['rep'],
                name='{0}_{1}-fold_{2}_rep'.format(base_name, k, variable),
                plot_text='{0} \pm {1}'.format(r_squared_avg, r_squared_std)
            )
            plot_predicted_actual(
                validation_dict[variable]['ind'], prediction_dict[variable]['ind'],
                name='{0}_{1}-fold_{2}_ind'.format(base_name, k, variable),
                plot_text='{0} \pm {1}'.format(r_squared_avg, r_squared_std)
            )
        else:
            plot_predicted_actual(
                validation_dict[variable], prediction_dict[variable],
                name='{0}_{1}-fold_{2}'.format(base_name, k, variable),
                plot_text='{0} \pm {1}'.format(r_squared_avg, r_squared_std)
            )
    sys.stdout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""trains a two layer perceptron regressor on input cross 
    validation data sets. models are made to predict fold induction, repression, induction and both repression and 
    induction together""")
    parser.add_argument('-f', '--features', default='pairwise',
                        help='features to be included in model. accepted values include:'
                             'single, pairwise')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001,)
    parser.add_argument('-n', '--nodes', type=int, default=100,
                        help='number of nodes in hidden layer')
    parser.add_argument('-a', '--activation', default='relu',
                        choices=['identity', 'logistic', 'tanh', 'relu'])
    parser.add_argument('-t', '--tolerance', type=float, default=0.0001,
                        help='tolerance for when system is considered converged')
    required = parser.add_argument_group('required')
    required.add_argument('-cv', '--cross_validation_files', required=True, nargs='*',
                          help='cross validation data files')
    args = parser.parse_args()
    perceptron_regression_cv(args.cross_validation_files, args.features,
                             args.nodes, args.verbose, args.learning_rate, args.activation, args.tolerance)
