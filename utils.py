import numpy as np
from noise_functions_binary import try_region_binary
import sys


def get_max(lst, target):
    # returns maximum of the list (ix, elt); omits target entry
    n1 = (-sys.maxint, None)
    for ix, elt in enumerate(lst):
        if ix == target:
            continue
        elif elt > n1[1]:
            n1 = (ix, elt)
    return n1


def find_noise_bounds_binary(models, X, Y):
    # find the min max bounds for each point given a list of models
    max_bounds = []
    num_models = len(models)
    for i in xrange(len(X)):
        max_r = -1 * Y[i] * np.ones(num_models)
        max_v = try_region_binary(models, max_r, X[i])
        max_bounds.append(np.linalg.norm(max_v))
    min_bounds = np.array([model.distance(X) for model in models]).T
    min_bounds = np.mean(min_bounds, axis=1)
    return min_bounds, max_bounds


def find_noise_bounds_multi(models, X):
    # find the minimum distance to the class boundary for each point in X
    min_bounds = np.array([model.distance(X) for model in models]).T
    min_bounds = np.mean(min_bounds, axis=1)
    return min_bounds


def jointly_shuffle_arrays(a, b, p=None):
    # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    assert len(a) == len(b)
    if p is None:
        p = np.random.permutation(len(a))
    return a[p], b[p], p


def generate_exp_data(num_pts, X, Y, models, target_dict=False, num_classes=4):
    # returns num_pts from (X, Y) that are correctly classified by all models
    num_selected = 0
    num_models = len(models)
    res_X = []
    res_Y = []
    for i in xrange(len(X)):
        all_correct = sum([model.evaluate(X[i:i+1], Y[i:i+1]) for model in models]) == num_models
        if all_correct:
            if target_dict:
                true_label = np.argmax(Y[i])
                target_labels = target_dict[true_label]
                for l in target_labels:
                    res_X.append(X[i])
                    res_Y.append((np.arange(num_classes) == l).astype(np.float32))
            else:
                res_X.append(X[i])
                res_Y.append(Y[i])
            num_selected += 1
        if num_selected == num_pts:
            break
    if num_selected < num_pts:
        print "Not enough points were correctly predicted by all models"
    return np.array(res_X), np.array(res_Y)


def subset_multiclass_data(data, labels, label_dict):
    # used for binary classification, subsets data to only include labels in label dict
    # label dict has the form of original label -> new label
    subset = set(label_dict.keys())
    X = []
    Y = []
    for i in xrange(len(data)):
        label = labels[i]
        if label in subset:
            label = label_dict[label]
            X.append(data[i])
            Y.append(label)
    return np.array(X), np.array(Y)


def generate_exp_data_dl(num_pts, models, X, Y):
    num_selected = 0
    res_X = []
    res_Y = []
    print len(X)
    for i in xrange(len(X)):
        if sum([model.evaluate(X[i:i + 1], Y[i:i + 1], verbose=0)[1] for model in models]) == len(models):
            res_X.append(X[i])
            res_Y.append(Y[i])
            num_selected += 1
        if num_selected == num_pts:
            break
    if num_selected != num_pts:
        print "not enough points were classified correctly"
    return np.array(res_X), np.array(res_Y)

