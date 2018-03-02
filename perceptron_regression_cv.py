#!/usr/bin/env python
import argparse
import numpy as np
import sys
from linear_regression_cv import cross_validation_dict_and_combos_from_txt, training_sets_from_cv_dict, \
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
        rep_ind_train = np.column_stack((rep_train, ind_train))
        rep_ind_validation = np.column_stack((rep_validation, ind_validation))
        sys.stdout.write('training for repression:\n')
        model_rep = MLPRegressor(
            hidden_layer_sizes=(nodes,),
            verbose=verbose,
            learning_rate_init=learning_rate,
            activation=activation,
            tol=tolerance,
        ).fit(x_train, rep_train)
        sys.stdout.write('training for induction:\n')
        model_ind = MLPRegressor(
            hidden_layer_sizes=(nodes,),
            verbose=verbose,
            learning_rate_init=learning_rate,
            activation=activation,
            tol=tolerance,
        ).fit(x_train, ind_train)
        sys.stdout.write('training for fold induction:\n')
        model_fi = MLPRegressor(
            hidden_layer_sizes=(nodes,),
            verbose=verbose,
            learning_rate_init=learning_rate,
            activation=activation,
            tol=tolerance,
        ).fit(x_train, fi_train)
        sys.stdout.write('training for both repression and induction:\n')
        model_rep_ind = MLPRegressor(
            hidden_layer_sizes=(nodes,),
            verbose=verbose,
            learning_rate_init=learning_rate,
            activation=activation,
            tol=tolerance,
        ).fit(x_train, rep_ind_train)

        y_hat_rep = model_rep.predict(x_validation)
        y_hat_ind = model_ind.predict(x_validation)
        y_hat_fi = model_fi.predict(x_validation)
        y_hat_rep_ind = model_rep_ind.predict(x_validation)

        r_squared_rep = model_rep.score(x_validation, rep_validation)
        r_squared_ind = model_ind.score(x_validation, ind_validation)
        r_squared_fi = model_fi.score(x_validation, fi_validation)
        r_squared_rep_ind = model_rep_ind.score(x_validation, rep_ind_validation)

        r_squared_dict['rep'].append(r_squared_rep)
        r_squared_dict['ind'].append(r_squared_ind)
        r_squared_dict['fi'].append(r_squared_fi)
        r_squared_dict['both'].append(r_squared_rep_ind)

        validation_dict['rep'].extend(rep_validation)
        validation_dict['ind'].extend(ind_validation)
        validation_dict['fi'].extend(fi_validation)
        validation_dict['both']['rep'].extend(rep_ind_validation[:, 0])
        validation_dict['both']['ind'].extend(rep_ind_validation[:, 1])

        prediction_dict['rep'].extend(y_hat_rep)
        prediction_dict['ind'].extend(y_hat_ind)
        prediction_dict['fi'].extend(y_hat_fi)
        prediction_dict['both']['rep'].extend(y_hat_rep_ind[:, 0])
        prediction_dict['both']['ind'].extend(y_hat_rep_ind[:, 1])

        sys.stdout.write('validation set {0}:\n'.format(missing_set))
        sys.stdout.write('r squared: rep: {0} ind: {1} fi: {2} both: {3} \n'.format(
            r_squared_rep, r_squared_ind, r_squared_fi, r_squared_rep_ind, ))

    sys.stdout.write(
        'avg r squared: rep: {0} +/- {1}\tind: {2} +/- {3}\tfi: {4} +/- {5}\tboth: {6} +/- {7}\n'.format(
            round(np.average(r_squared_dict['rep']), 2), round(np.std(r_squared_dict['rep']), 2),
            round(np.average(r_squared_dict['ind']), 2), round(np.std(r_squared_dict['ind']), 2),
            round(np.average(r_squared_dict['fi']), 2), round(np.std(r_squared_dict['fi']), 2),
            round(np.average(r_squared_dict['both']), 2), round(np.std(r_squared_dict['both']), 2),
        )
    )
    base_name = cross_validation_files[0].split('/')[-1].split('_cv_')[0]
    plot_predicted_actual(
        validation_dict['rep'], prediction_dict['rep'],
        name='{0}_{1}-fold_rep'.format(base_name, k),
        plot_text='{0} \pm {1}'.format(
            round(np.average(r_squared_dict['rep']), 2), round(np.std(r_squared_dict['rep']), 2)
        ),
    )
    plot_predicted_actual(
        validation_dict['ind'], prediction_dict['ind'],
        name='{0}_{1}-fold_ind'.format(base_name, k),
        plot_text='{0} \pm {1}'.format(
            round(np.average(r_squared_dict['ind']), 2), round(np.std(r_squared_dict['ind']), 2)
        ),
    )
    plot_predicted_actual(
        validation_dict['fi'], prediction_dict['fi'],
        name='{0}_{1}-fold_fi'.format(base_name, k),
        plot_text='{0} \pm {1}'.format(
            round(np.average(r_squared_dict['fi']), 2), round(np.std(r_squared_dict['fi']), 2)
        ),
    )
    plot_predicted_actual(
        validation_dict['both']['rep'], prediction_dict['both']['rep'],
        name='{0}_{1}-fold_both_rep'.format(base_name, k),
        plot_text='{0} \pm {1}'.format(
            round(np.average(r_squared_dict['both']), 2), round(np.std(r_squared_dict['both']), 2)
        )
    )
    plot_predicted_actual(
        validation_dict['both']['ind'], prediction_dict['both']['ind'],
        name='{0}_{1}-fold_both_ind'.format(base_name, k),
        plot_text='{0} \pm {1}'.format(
            round(np.average(r_squared_dict['both']), 2), round(np.std(r_squared_dict['both']), 2)
        )
    )


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
