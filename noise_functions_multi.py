import numpy as np
from cvxopt import matrix, solvers
from itertools import product
from functools import partial
import sys
import time
import ray


#TODO: add remote decorator to this, make the oracle parallel!
def try_region_multi(models, labels, x, delta=1e-10):
    P = matrix(np.identity(x.shape[0]))
    q = matrix(np.zeros(x.shape[0]))
    h = []
    G = []
    num_models = len(models)
    for i in xrange(num_models):
        others = range(10)
        target = labels[i]
        del others[target]
        target_w, target_b = models[i].weights[target], models[i].bias[target]
        for j in others:
            other_w, other_b = models[i].weights[j], models[i].bias[j]
            ineq_val = np.dot(target_w - other_w, x) + target_b - other_b - delta
            h.append(ineq_val)
            G.append(other_w - target_w)
    h = matrix(h)
    G = matrix(np.array(G))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    if sol['status'] == 'optimal':
        v = np.array(sol['x']).reshape(-1,)
        perturbed_x = np.array(x + v).reshape(1, -1)
        is_desired_label = [models[i].predict(perturbed_x)[0] == labels[i] for i in xrange(num_models)]
        if sum(is_desired_label) == num_models:
            return v
        else:
            return try_region_multi(models, labels, x, delta * 1.5)
    else:
        return None


def distributional_oracle_multi(distribution, models, x, y, alpha, target=False):
    num_models = len(models)

    labels_values = []
    for labels in product(range(10), repeat=num_models):  # iterate over all possible regions
        if target:
            is_misclassified = (np.array(labels) == target).astype(np.float32)
        else:
            is_misclassified = (np.array(labels) != y).astype(np.float32)
        value = np.dot(is_misclassified, distribution)
        labels_values.append((labels, value))

    values = sorted(set([value for label, value in labels_values]), reverse=True)

    for curr_value in values:
        feasible_candidates = []
        for labels in [labels for labels, val in labels_values if val == curr_value]:
            v = try_region_multi(models, labels, x)
            if v is not None:
                norm = np.linalg.norm(v)
                if norm <= alpha:
                    feasible_candidates.append((v, norm))
        # amongst those with the max value, return the one with the minimum norm
        if feasible_candidates:
            # break out of the loop since we have already found the optimal answer
            return min(feasible_candidates, key=lambda x: x[1])[0]
    return np.zeros(x.shape[0])  # we can't trick anything


@ray.remote
def grad_desc_targeted(distribution, models, x, target, alpha, learning_rate=.001,
                       iters=3000, early_stop=5, box_min=0.0, box_max=1.0):
    v = np.zeros(len(x))
    best_sol = (sys.maxint, v)
    loss_queue = []
    for i in xrange(iters):
        gradient = sum([-1 * p * model.gradient(np.array([x + v]), [target])
                        for p, model in zip(distribution, models)])[0]

        v += learning_rate * gradient

        # clip values so they lie in the appropriate range
        curr_sol = np.clip(x + v, box_min, box_max)
        v = curr_sol - x

        norm = np.linalg.norm(v)
        if norm >= alpha:
            v = v / norm * alpha

        loss = np.dot(distribution, [model.rhinge_loss([x + v], [target])[0] for model in models])
        loss_queue = [loss] + loss_queue

        if i >= early_stop:
            del loss_queue[-1]
            val = loss_queue[-1]
            if sum([val == q_val for q_val in loss_queue]) == early_stop:
                break

        if loss < best_sol[0]:
            best_sol = (loss, v)

        if loss == 0:
            break
    return best_sol


@ray.remote
def grad_desc_convex(distribution, models, x, y, alpha, target=False, learning_rate=.001, iters=3000, early_stop=5,
                     box_min=0.0, box_max=1.0):
    if target:
        return grad_desc_targeted(distribution, models, x, target, alpha, learning_rate, iters, early_stop, box_min,
                                  box_max)[1]
    else:
        other_labels = range(10)
        del other_labels[y]
        best_sol = (sys.maxint, None)
        for label in other_labels:
            sol = grad_desc_targeted.remote(distribution, models, x, label, alpha, learning_rate, iters, early_stop,
                                            box_min, box_max)
            sol = ray.get(sol)
            if sol[0] < best_sol[0]:
                best_sol = sol
            if best_sol[0] == 0.0:
                return best_sol[1]
        return best_sol[1]


@ray.remote
def grad_desc_nonconvex(distribution, models, x, y, alpha, learning_rate=.001, iters=3000, early_stop=5,
                        box_min=0.0, box_max=1.0):
    v = np.zeros(len(x))
    best_sol = (sys.maxint, v)
    loss_queue = []
    for i in xrange(iters):
        gradient = sum([-1 * p * model.gradient_untargeted(np.array([x + v]), [y])
                        for p, model in zip(distribution, models)])[0]
        v += learning_rate * gradient

        # clip values so they lie in the appropriate range
        curr_sol = np.clip(x + v, box_min, box_max)
        v = curr_sol - x

        norm = np.linalg.norm(v)
        if norm >= alpha:
            v = v / norm * alpha

        loss = np.dot(distribution, [model.untargeted_loss(np.array([x + v]), [y])[0] for model in models])

        loss_queue = [loss] + loss_queue
        if i >= early_stop:
            del loss_queue[-1]
            val = loss_queue[-1]
            if sum([val == q_val for q_val in loss_queue]) == early_stop:
                break

        if loss < best_sol[0]:
            best_sol = (loss, v)

        if loss == 0:
            break
    return best_sol[1]


FUNCTION_DICT_MULTI = {"oracle": distributional_oracle_multi,
                       "grad_desc_convex": grad_desc_convex,
                       "grad_desc_nonconvex": grad_desc_nonconvex}

