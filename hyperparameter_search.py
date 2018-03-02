#!/usr/bin/env python
import argparse
import itertools
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from linear_regression_cv import cross_validation_dict_and_combos_from_txt, training_sets_from_cv_dict


def hyperparameter_search(input_files, estimator, base_name, verbose):
    if not base_name:
        base_name = '{0}'.format(estimator)
    k = len(input_files)
    cross_validation_dict, cross_validation_combos = cross_validation_dict_and_combos_from_txt(input_files)
    x_train, rep_train, ind_train, fi_train = training_sets_from_cv_dict(range(k), cross_validation_dict)
    training_dict = {
        'rep': rep_train,
        'ind': ind_train,
        'fi': fi_train,
        'both': np.column_stack((rep_train, ind_train))
    }

    if estimator == 'perceptron':
        regressor = MLPRegressor()
        layer_sizes = [10, 20, 50, 100, 200, 500, 1000]
        single_layers = list(itertools.product(layer_sizes))
        double_layers = list(itertools.product(layer_sizes, layer_sizes))
        param_grid = {
            'hidden_layer_sizes': single_layers + double_layers,
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['adam'],
            'alpha': np.logspace(-3, -5, 10),
            'learning_rate_init': np.logspace(-2, -4, 10),
            'tol': np.logspace(-4, -6, 10)
        }
    elif estimator == 'decision_tree':
        regressor = DecisionTreeRegressor()
        param_grid = {
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'max_depth': [None,],
            'min_samples_split': range(2, 6),
            'min_samples_leaf': range(1, 6),
        }
    elif estimator == 'ridge':
        regressor = Ridge()
        param_grid = {
            'alpha': np.logspace(-6, 4, 100),
            'solver': ['auto', 'svd', 'choelsky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'tol': np.logspace(-2, -4, 10)
        }
    elif estimator == 'lasso':
        regressor = Lasso()
        param_grid = {
            'alpha': np.logspace(-4, 1, 100),
            'tol': np.logspace(-2, -4, 10)
        }
    else:
        raise Exception('an allowable estimator was not given as input')

    for variable in ['rep', 'ind', 'fi', 'both']:
        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=10, verbose=verbose)
        grid_search.fit(x_train, training_dict[variable])
        gs_dataframe = pd.DataFrame.from_dict(grid_search.cv_results_)
        gs_dataframe.to_csv('{0}.csv'.format(base_name))
        sys.stdout.write('{0}\nbest score: {1}\nbest params: {2}\nbest index: {3}\n'.format(
            variable, grid_search.best_score_, grid_search.best_params_, grid_search.best_index_))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""perform exhaustive search over specificed parameter 
    values for an estimator. currently the parameter values for each allowable estimator are hard coded.
     input is space separated txt file(s) with data, the estimator to be used, """)
    parser.add_argument('-n', '--name', help='name of output csv file')
    parser.add_argument('-v', '--verbose', type=int, default=0, choices=[0, 1, 2, 3])
    required = parser.add_argument_group('required')
    required.add_argument('-i', '--input_files', nargs='*', required=True,
                          help='input file(s) containing data that will be used to tune hyperparameters')
    required.add_argument('-e', '--estimator', required=True,
                          choices=['ridge', 'lasso', 'perceptron', 'decision_tree'],
                          help='sklearn regression estimator')
    args = parser.parse_args()
    hyperparameter_search(args.input_files, args.estimator, args.name, args.verbose)
