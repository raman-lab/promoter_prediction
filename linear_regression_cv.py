#!/usr/bin/env python
import argparse
import itertools
import numpy as np
import sys
from encode_one_hot import promoter_data_file_to_lists, encode_one_hot
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV, LinearRegression


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

    # for now i am not going to use plotly, may revisit
    # elif library == 'plotly':
    #     import plotly
    #     import plotly.graph_objs as go
    #
    #     trace_scatter = go.Scatter(
    #         x=actual,
    #         y=predicted,
    #         mode='markers',
    #         marker=dict(color=almost_gray)
    #     )
    #     trace_line = go.Scatter(
    #         x=[min_y, max_y],
    #         y=[min_y, max_y],
    #         mode='lines',
    #         marker=dict(color=almost_black)
    #     )
    #
    #     fig = {
    #         'data': [trace_line, trace_scatter],
    #         'layout': {
    #             'xaxis': {'title': 'Actual'},
    #             'yaxis': {'title': 'Predicted'},
    #             'title': name}
    #     }
    #     plotly.offline.plot(fig, filename='{0}.html'.format(name), auto_open=False)


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

    if not alpha:
        x_train, rep_train, ind_train, fi_train = training_sets_from_cv_dict(range(k), cross_validation_dict)
        if model == 'lasso':
            model_rep = LassoCV(cv=10).fit(x_train, rep_train)
            model_ind = LassoCV(cv=10).fit(x_train, ind_train)
            model_fi = LassoCV(cv=10).fit(x_train, fi_train)
            alpha_rep = model_rep.alpha_
            alpha_ind = model_ind.alpha_
            alpha_fi = model_fi.alpha_
        elif model == 'ridge':
            model_rep = RidgeCV(alphas=np.logspace(-4, 3), cv=10).fit(x_train, rep_train)
            model_ind = RidgeCV(alphas=np.logspace(-4, 3), cv=10).fit(x_train, ind_train)
            model_fi = RidgeCV(alphas=np.logspace(-4, 3), cv=10).fit(x_train, fi_train)
            alpha_rep = model_rep.alpha_
            alpha_ind = model_ind.alpha_
            alpha_fi = model_fi.alpha_
        elif model == 'linreg':
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

    r_squared_dict = {'rep': [], 'ind': [], 'fi': []}
    validation_dict = {'rep': [], 'ind': [], 'fi': []}
    prediction_dict = {'rep': [], 'ind': [], 'fi': []}

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

        if model == 'lasso':
            model_rep = Lasso(alpha=alpha_rep).fit(x_train, rep_train)
            model_ind = Lasso(alpha=alpha_ind).fit(x_train, ind_train)
            model_fi = Lasso(alpha=alpha_fi).fit(x_train, fi_train)
        elif model == 'ridge':
            model_rep = Ridge(alpha=alpha_rep).fit(x_train, rep_train)
            model_ind = Ridge(alpha=alpha_ind).fit(x_train, ind_train)
            model_fi = Ridge(alpha=alpha_fi).fit(x_train, fi_train)
        elif model == 'linreg':
            model_rep = LinearRegression().fit(x_train, rep_train)
            model_ind = LinearRegression().fit(x_train, ind_train)
            model_fi = LinearRegression().fit(x_train, fi_train)
        else:
            raise Exception('regression method not specified')

        y_hat_rep = model_rep.predict(x_validation)
        y_hat_ind = model_ind.predict(x_validation)
        y_hat_fi = model_fi.predict(x_validation)
        r_squared_rep = model_rep.score(x_validation, rep_validation)
        r_squared_ind = model_ind.score(x_validation, ind_validation)
        r_squared_fi = model_fi.score(x_validation, fi_validation)
        r_squared_dict['rep'].append(r_squared_rep)
        r_squared_dict['ind'].append(r_squared_ind)
        r_squared_dict['fi'].append(r_squared_fi)
        validation_dict['rep'].extend(rep_validation)
        validation_dict['ind'].extend(ind_validation)
        validation_dict['fi'].extend(fi_validation)
        prediction_dict['rep'].extend(y_hat_rep)
        prediction_dict['ind'].extend(y_hat_ind)
        prediction_dict['fi'].extend(y_hat_fi)

        sys.stdout.write('validation set {0}:\n'.format(missing_set))
        sys.stdout.write('r squared: rep: {0}\tind: {1}\tfi: {2}\n'.format(r_squared_rep, r_squared_ind, r_squared_fi))
        sys.stdout.write('non zero coefs: rep: {0}\tind: {1}\tfi: {2}\n'.format(
            np.count_nonzero(model_rep.coef_), np.count_nonzero(model_ind.coef_),
            np.count_nonzero(model_fi.coef_)
        ))
    sys.stdout.write('avg r squared: rep: {0} +/- {1}\tind: {2} +/- {3}\tfi: {4} +/- {5}\n'.format(
        round(np.average(r_squared_dict['rep']), 2), round(np.std(r_squared_dict['rep']), 2),
        round(np.average(r_squared_dict['ind']), 2), round(np.std(r_squared_dict['ind']), 2),
        round(np.average(r_squared_dict['fi']), 2), round(np.std(r_squared_dict['fi']), 2),
    ))
    base_name = cross_validation_files[0].split('/')[-1].split('_cv_')[0]
    plot_predicted_actual(
        validation_dict['rep'], prediction_dict['rep'],
        name='{0}_{1}_{2}-fold_rep'.format(base_name, model, k),
        plot_text='{0} \pm {1}'.format(
            round(np.average(r_squared_dict['rep']), 2), round(np.std(r_squared_dict['rep']), 2)),
    )
    plot_predicted_actual(
        validation_dict['ind'], prediction_dict['ind'],
        name='{0}_{1}_{2}-fold_ind'.format(base_name, model, k),
        plot_text='{0} \pm {1}'.format(
            round(np.average(r_squared_dict['ind']), 2), round(np.std(r_squared_dict['ind']), 2)),
    )
    plot_predicted_actual(
        validation_dict['fi'], prediction_dict['fi'],
        name='{0}_{1}_{2}-fold_fi'.format(base_name, model, k),
        plot_text='{0} \pm {1}'.format(
            round(np.average(r_squared_dict['fi']), 2), round(np.std(r_squared_dict['fi']), 2)),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='runs linear regression using either Lasso or Ridge on input cross validation data sets. single '
                    'and pairwise nucleotides are used as features. regression models are made to predict repression '
                    'and induction values, assumed to be columns 2 and 3, respectively. if alpha is not provided, '
                    'it will be selected through error minimization with cross validation')
    parser.add_argument('-a', '--alpha', type=float, help='alpha parameter used in regularization by Lasso and Ridge')
    parser.add_argument('-m', '--model', choices=['linreg', 'lasso', 'ridge'], default='linreg',
                        help='type of linear regression model to use')

    required = parser.add_argument_group('required')
    required.add_argument('-cv', '--cross_validation_files', required=True, nargs='*',
                          help='cross validation data files')
    args = parser.parse_args()
    linear_regression_cv(args.cross_validation_files, args.model, args.alpha)
