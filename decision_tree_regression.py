#!/usr/bin/env python
import argparse
import numpy as np
import sys
from sklearn.tree import DecisionTreeRegressor
from linear_regression_cv import cross_validation_dict_and_combos_from_txt, data_arrays_from_cv_dict, \
    plot_predicted_actual


def descision_tree_regression(cross_validation_files, criterion):
    k = len(cross_validation_files)
    cross_validation_dict, cross_validation_combos = \
        cross_validation_dict_and_combos_from_txt(cross_validation_files)

    r_squared_dict = {'rep': [], 'ind': [], 'fi': [], 'both': []}
    validation_dict = {'rep': [], 'ind': [], 'fi': [], 'both': {'rep': [], 'ind': []}}
    prediction_dict = {'rep': [], 'ind': [], 'fi': [], 'both': {'rep': [], 'ind': []}}

    for cv_combo in cross_validation_combos:
        x_train, rep_train, ind_train, fi_train = data_arrays_from_cv_dict(cv_combo, cross_validation_dict)
        missing_set = list(set(range(k)) - set(cv_combo))[0]
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
        sys.stdout.write('validation set {0}:\n r squared: '.format(missing_set))

        for variable in ['rep', 'ind', 'fi', 'both']:
            regressor = DecisionTreeRegressor()
            regressor.fit(x_train, training_dict[variable])
            y_hat = regressor.predict(x_valid)
            r_squared = regressor.score(x_valid, valid_dict[variable])
            if variable == 'both':
                validation_dict[variable]['rep'].extend(valid_dict[variable][:, 0])
                validation_dict[variable]['ind'].extend(valid_dict[variable][:, 1])
                prediction_dict[variable]['rep'].extend(y_hat[:, 0])
                prediction_dict[variable]['ind'].extend(y_hat[:, 1])
            else:
                validation_dict[variable].extend(validation_dict[variable])
                prediction_dict[variable].extend(y_hat)
            sys.stdout.write('{0}: {1}\t'.format(variable, r_squared))
        sys.stdout.write('\n')

    base_name = cross_validation_files[0].split('/')[-1].split('_cv_')[0]
    sys.stdout.write('avg r sqaured: ')
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
    parser = argparse.ArgumentParser(description="""trains a decision regressor on input cross 
    validation data sets. models are made to predict fold induction, repression, 
    induction and both repression and induction together. 
    currently tree pruning is not implemented, so trees will likely be overfit""")
    parser.add_argument('-c', '--criterion', default='mse', choices=['mse', 'friedman_mse', 'mae'],
                        help='function used to measure the quality of a split')
    required = parser.add_argument_group('required')
    required.add_argument('-cv', '--cross_validation_files', required=True, nargs='*',
                          help='cross validation data files')
    args = parser.parse_args()
    descision_tree_regression(args.cross_validation_files, args.criterion)
