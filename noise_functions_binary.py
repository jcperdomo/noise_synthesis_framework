from functools import partial
from cvxopt import matrix, solvers
from itertools import product
import numpy as np


def try_region_binary(models, signs, x, delta=1e-10):
    """
    models: list of LinearBinaryClassifiers
    signs: list of signs (+1, -1) of length num_models
    x: np array of shape dim (a single point)
    returns: a vector in the region denoted by the signs vector
    """
    dim = x.shape[0]
    P = matrix(np.identity(dim))
    q = matrix(np.zeros(dim))
    h = []
    G = []
    num_models = len(models)
    for i in xrange(num_models):
        weights, bias = models[i].weights.T, models[i].bias
        ineq_val = -1.0 * delta + signs[i] * (np.dot(weights, x) + bias)
        h.append(ineq_val[0])
        G.append(-1.0 * signs[i] * weights.reshape(-1,))
    h = matrix(h)
    G = matrix(np.array(G))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    if sol['status'] == 'optimal':
        v = np.array(sol['x']).reshape(-1,)
        perturbed_x = np.array(x + v).reshape(1, -1)
        is_desired_sign = [models[i].predict(perturbed_x)[0] == signs[i] for i in xrange(num_models)]
        if sum(is_desired_sign) == num_models:
            return v
        else:
            return try_region_binary(models, signs, x, delta * 1.5)
    else:
        return None


def distributional_oracle_binary(distribution, models, x, y, alpha):
    """
    computes the optimal perturbation for x under alpha and the given distribution
    """
    num_models = len(models)
    # we should only take into consideration models that we could feasibly trick
    dists = [model.distance(x) for model in models]
    feasible_models = [models[i] for i in xrange(num_models) if dists[i] < alpha]
    distribution = np.array([distribution[i] for i in xrange(num_models) if dists[i] < alpha])
    num_models = len(feasible_models)

    # can't trick anything
    if num_models == 0:
        return np.zeros(x.shape)

    signs_values = []
    for signs in product([-1.0, 1.0], repeat=num_models):  # iterate over all possible regions
        is_misclassified = np.equal(-1.0 * y * np.ones(num_models), signs)  # y = -1, or 1
        value = np.dot(is_misclassified, distribution)
        signs_values.append((signs, value))

    values = sorted(set([value for signs, value in signs_values]), reverse=True)
    for value in values:
        feasible_candidates = []
        for signs in [sign for sign, val in signs_values if val == value]:
            v = try_region_binary(feasible_models, signs, x)
            if v is not None:
                norm = np.linalg.norm(v)
                if norm <= alpha:
                    feasible_candidates.append((v, norm))
        # amongst those with the max value, return the one with the minimum norm
        if feasible_candidates:
            # break out of the loop since we have already found the optimal answer
            return min(feasible_candidates, key=lambda x: x[1])[0]


def grad_desc_binary(distribution, models, x, y, alpha, learning_rate=.001, T=3000):
    v = np.zeros(len(x))
    for i in xrange(T):
        loss = np.dot(distribution, [model.rhinge_loss(x + v, y) for model in models])
        if loss == 0:
            break

        gradient = sum([-1 * w * model.gradient(x + v, y) for w, model in zip(distribution, models)])[0]
        v += learning_rate * gradient
        norm = np.linalg.norm(v)
        if norm >= alpha:
            v = v / norm * alpha
    return v


FUNCTION_DICT_BINARY = {"oracle": distributional_oracle_binary, "gradientDescent": grad_desc_binary}


