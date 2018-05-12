import numpy as np
import time
import logging as log
import ray


def evaluate_costs(models, V, X, Y, targets, dl=False):
    """
    Returns the 0-1 loss of the models on input (X + V, Y) if targets is False
    Else returns the target accuracy of the models on (X + V, Targets)
    dl is a bool to indicate whether the models are linear classifiers or keras deep learning models
    """
    if targets is not False:
        if dl:
            res = np.array([model.evaluate(X + V, targets)[1] for model in models])
        else:
            res = np.array([model.evaluate(X + V, targets) for model in models])
    else:
        if dl:
            res = np.array([1 - model.evaluate(X + V, Y)[1] for model in models])
        else:
            res = np.array([1 - model.evaluate(X + V, Y) for model in models])
    return res


def adversary(distribution, models, X, Y, alpha, noise_func, targets, use_ray=True):
    """
    uses the noise function to compute adversarial perturbations that maximize the loss of the learner under
    the chosen distribution
    """
    if use_ray:
        noise_func = noise_func.remote

    if targets is not False:
        res = [noise_func(distribution, models, x, y, alpha, target=target) for x, y, target
               in zip(X, Y, targets)]

    else:
        res = [noise_func(distribution, models, x, y, alpha) for x, y in zip(X, Y)]

    if use_ray:
        res = ray.get(res)

    return np.array(res)


def run_mwu(models, iters, X, Y, alpha, noise_func, epsilon=None, targeted=False, dl=False, use_ray=True):

    if dl:
        ray.init(num_gpus=4)
    else:
        ray.init()


    num_models = len(models)

    # compute epsilon as a function of the number of rounds, see paper for more details
    if epsilon is None:
        delta = np.sqrt(4 * np.log(num_models) / float(iters))
        epsilon = delta / 2.0
    else:
        delta = 2.0 * epsilon

    log.info("\nRunning MWU for {} Iterations with Epsilon {}\n".format(iters, epsilon))
    log.info("Guaranteed to be within {} of the minimax value \n".format(delta))

    loss_history = []
    costs = []
    acc_history = []
    v = []
    w = []
    action_loss = []

    w.append(np.ones(num_models) / num_models)

    for t in xrange(iters):

        log.debug("Iteration {}\n".format(t))

        start_time = time.time()

        v_t = adversary(w[t], models, X, Y, alpha, noise_func, targeted, use_ray)
        v.append(v_t)

        cost_t = evaluate_costs(models, v_t, X, Y, targeted, dl=dl)
        costs.append(cost_t)

        if targeted is not False:
            avg_loss = np.mean((np.array(costs)), axis=0)
            min_loss = min(avg_loss)
            acc_history.append(min_loss)
        else:
            avg_acc = np.mean((1 - np.array(costs)), axis=0)
            max_acc = max(avg_acc)
            acc_history.append(max_acc)

        loss = np.dot(w[t], cost_t)
        individual = [w[t][j] * cost_t[j] for j in xrange(num_models)]

        log.debug("Weights {} Sum of Weights {}".format(w[t], sum(w[t])))

        if targeted is not False:
            log.debug("Minimum (Average) Loss of Classifier {}".format(acc_history[-1]))
            if dl:
                log.debug("Cost (Before Noise) {}".format(np.array([model.evaluate(X, targeted, verbose=0)[1] for model in models])))
            else:
                log.debug("Cost (Before Noise) {}".format(np.array([model.evaluate(X, targeted) for model in models])))

        else:
            log.debug("Maximum (Average) Accuracy of Classifier {}".format(acc_history[-1]))
            if dl:
                log.debug("Cost (Before Noise) {}".format(np.array([1 - model.evaluate(X, Y, verbose=0)[1] for model in models])))
            else:
                log.debug("Cost (Before Noise) {}".format(np.array([1 - model.evaluate(X, Y) for model in models])))

        log.debug("Cost (After Noise), {}".format(cost_t))
        log.debug("Loss {} Loss Per Action {}".format(loss, individual))

        loss_history.append(loss)
        action_loss.append(individual)

        new_w = np.copy(w[t])

        # penalize experts
        for i in xrange(num_models):
            new_w[i] *= (1.0 - epsilon) ** cost_t[i]

        # renormalize weights
        w_sum = new_w.sum()
        for i in xrange(num_models - 1):
            new_w[i] = new_w[i] / w_sum
        new_w[-1] = 1.0 - new_w[:-1].sum()

        w.append(new_w)

        log.debug("time spent {}\n".format(time.time() - start_time))
    log.info("finished running MWU ")
    return w, v, loss_history, acc_history, action_loss
