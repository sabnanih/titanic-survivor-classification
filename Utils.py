import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression
from sklearn.model_selection import learning_curve, validation_curve

def plot_curve(Xlist, Ylist, title, xlabel, ylabel, plotlabels):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    cnt = 0
    for X in Xlist:
        Y = Ylist[cnt]
        label = plotlabels[cnt]
        plt.plot(X, Y, 'o-', label=label)
        cnt += 1
    plt.legend(loc="best")
    plt.show()
    plt.clf()

def load_dataset():
    train_initial = pd.read_csv('datasets/train.csv')
    test_initial = pd.read_csv('datasets/test.csv')
    return train_initial, test_initial

def write_to_file(df, filename='file.csv', sep=','):
    df.to_csv(filename, sep=sep, index=False)

def preprocess(train, impute_data=False, normalize_data=False, is_train=True):
    if is_train:
        # ignore features that are not useful
        train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        numeric_features = ['Survived', 'Age', 'SibSp', 'Parch', 'Fare']
    else:
        train = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
        numeric_features = ['PassengerId', 'Survived', 'Age', 'SibSp', 'Parch', 'Fare']

    # get non-numeric columns
    non_numeric_columns = np.setdiff1d(train.columns.values, numeric_features)

    # create dummy variables for categorical variables
    # TODO: ideally would want to create these based on a list of possible feature values,
    # otherwise will have to combine with test data and then create dummy variables
    # as some feature values might be present in one set but not the other sample
    train = pd.get_dummies(train, columns=non_numeric_columns, drop_first=True, dummy_na=True)

    train = train.drop(['Pclass_nan', 'Sex_nan'], axis=1)

    numeric_features_except_label = numeric_features[1:]

    # replace missing values with mean. ideally we will want to do this separately in each training fold separately
    # but here we are focusing on training only and not validation / leakage
    if impute_data:
        train = train.fillna(train.mean())
    if normalize_data:
        train[numeric_features_except_label] = (train[numeric_features_except_label] - train[numeric_features_except_label].min()) / \
                                               (train[numeric_features_except_label].max() - train[
                                                   numeric_features_except_label].min())
    if is_train:
        train_X = train.drop(['Survived'], axis=1).values
        train_y = train[['Survived']].values
    else:
        train_X = train.drop(['PassengerId'], axis=1).values
        train_y = train[['PassengerId']].values
    return train_X, train_y

def get_multiple_estimates(X, y, learning_rate=[0.0000000001], max_iter=1000, iteration_threshold=100, plotlabels=None,
                           reg_strength=0, regularization="Ridge", cost_threshold=None):
    cost_by_lr = []
    iterations = []
    plotlabels = []

    for lr in learning_rate:
        estimator_logReg = LogisticRegression(learning_rate=lr, reg_strength=reg_strength,
                                              regularization=regularization,
                                              max_iter=max_iter, iteration_threshold=iteration_threshold, cost_threshold=cost_threshold)
        estimator_logReg.fit(X, y)

        cost_by_lr.append(estimator_logReg.cost_by_iteration.tolist())
        iterations.append(estimator_logReg.iterations.tolist())
        plotlabels.append("Learning rate = " + str(lr))

    return cost_by_lr, iterations, plotlabels

# code re-used from scikit-learn example
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 2),
                        scoring='neg_mean_squared_error'):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score (" + str(scoring) + ")")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
                                                            scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    plt.clf()

# code re-used from scikit-learn example
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None, scoring='neg_mean_squared_error',
                          plot_log_scale=True, xlabel="Regularization parameter"):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.ylabel("Score (" + str(scoring) + ")")
    if ylim is not None:
        plt.ylim(*ylim)
    lw = 2
    if plot_log_scale:
        plt.xlabel(xlabel + " (Log Scale)")
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    else:
        plt.xlabel(xlabel)
        plt.plot(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)

    if plot_log_scale:
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
    else:
        plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    plt.clf()