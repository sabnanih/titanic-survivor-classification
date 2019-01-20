from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import numpy as np

class NaiveBayes(BaseEstimator):

    def __init__(self, laplace_smoothing=1e-10):
        self.laplace_smoothing=laplace_smoothing

    def fit(self, X, y):
        """
        fits Naive Bayes on labeled dataset <X, y>
        :param X: feature matrix
        :param y: labeled value
        :return: self
        """

        clf_gaussian = GaussianNB(var_smoothing=0)
        clf_bernoulli = BernoulliNB(alpha=self.laplace_smoothing)

        X_bernoulli, X_gaussian = self._partition_data(X)

        clf_gaussian.fit(X_gaussian, y)
        clf_bernoulli.fit(X_bernoulli, y)

        self._class_prior_ = clf_gaussian.class_prior_
        self._theta = clf_gaussian.theta_
        self._sigma = clf_gaussian.sigma_
        self._feature_log_prob = clf_bernoulli.feature_log_prob_

        return self

    def predict(self, X):
        X_bernoulli, X_gaussian = self._partition_data(X)
        # use math trick to compute everything without if statements: px + (1-x)(1-p) = 2px + 1-p-x = x(p-(1-p)) + (1-p)
        neg_log_prob = np.log(1 - np.exp(self._feature_log_prob))
        bernoulli_log_prob = np.log(self._class_prior_) + np.dot(X_bernoulli, (self._feature_log_prob - neg_log_prob).T) + neg_log_prob.sum(axis=1)

        gaussian_log_prob = []
        for i in range(np.size(self._class_prior_)):
            log_prob_ij = -0.5 * np.sum(np.log(2. * np.pi * self._sigma[i, :]))
            log_prob_ij -= 0.5 * np.sum(((X_gaussian - self._theta[i, :]) ** 2) /
                                 (self._sigma[i, :]), 1)
            gaussian_log_prob.append(log_prob_ij)

        gaussian_log_prob = np.array(gaussian_log_prob).T

        combined_log_prob = np.add(bernoulli_log_prob, gaussian_log_prob)
        y = np.array(np.argmax(combined_log_prob, axis=1))
        return y

    # partition data by numeric and non-numeric features: only checks if min value is 0 and max is 1 -
    # would classify continuous feature as binary if there are fractional values in between 0 and 1 in the data
    # TODO: fix the functionality
    def _partition_data(self, X):
        zero_val = np.array([0 for i in range(0, X.shape[1])])
        one_val = np.array([1 for i in range(0, X.shape[1])])

        min_val = np.amin(X, axis=0)
        max_val = np.amax(X, axis=0)

        bernoulli_idx = np.logical_and(np.logical_or(zero_val == min_val, one_val == min_val),
                                       np.logical_or(one_val == max_val, zero_val == max_val))

        return X[:, bernoulli_idx], X[:, np.logical_not(bernoulli_idx)]
