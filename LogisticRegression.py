from sklearn.base import BaseEstimator
import numpy as np
import time

class LogisticRegression(BaseEstimator):

    def __init__(self, learning_rate=0.001, reg_strength=0, regularization="Ridge", max_iter=1000, cost_threshold=None,
                 iteration_threshold=100):
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.regularization = regularization
        self.cost_threshold = cost_threshold
        self.max_iter = max_iter
        self.iteration_threshold = iteration_threshold

    def fit(self, X, y):
        """
        fits Logistic Regression using GD on labeled dataset <X, y>
        :param X: feature matrix
        :param y: labeled value
        :return: self
        """
        start_time = time.time()

        num_examples = X.shape[0]
        num_features = X.shape[1]
        theta = np.zeros((1, num_features))
        bias = 0
        g = bias + np.dot(X,theta.T)
        h = self._sigmoid(g)
        prev_cost = -1
        curr_cost = -(1/num_examples) * sum(y*np.log(h) + (1-y)*np.log(1-h))

        iter = 0
        cost = curr_cost.copy()
        iterations = np.array(iter)
        theta_by_iteration = []
        lr = self.learning_rate

        while self.cost_threshold is None or prev_cost < 0 or (prev_cost - curr_cost) > self.cost_threshold:
            prev_cost = curr_cost
            theta = theta - (1 / num_examples) * lr * (sum((h - y) * X) + self.reg_strength * theta)
            bias = bias - (1 / num_examples) * lr * sum((h - y))
            g = bias + np.dot(X, theta.T)
            h = self._sigmoid(g)
            curr_cost = -(1/num_examples) * sum(y*np.log(h) + (1-y)*np.log(1-h))

            iter += 1
            if iter % self.iteration_threshold == 0:
                cost = np.append(cost, [curr_cost])
                iterations = np.append(iterations, [iter])
                theta_by_iteration.append(theta.tolist())

            if self.cost_threshold is None and iter >= self.max_iter:
                break

        end_time = time.time()

        self.training_time = end_time - start_time
        self.weight_by_iteration = theta_by_iteration
        self.cost_by_iteration = cost
        self.iterations = iterations
        self.final_cost = curr_cost
        self._intercept = bias
        self._coef = theta

        return self

    def predict(self, X):
        g = self._intercept + np.dot(X, self._coef.T)
        h = self._sigmoid(g)
        y = np.array([1 if x >= 0.5 else 0 for x in np.nditer(h)])
        return y

    def _sigmoid(self, g):
        return 1/(1+np.exp(g*(-1)))
