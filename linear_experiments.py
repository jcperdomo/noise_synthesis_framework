import logging as log
import argparse
import os
import numpy as np
from linear_models import LinearBinaryClassifier, LinearOneVsAllClassifier
from noise_functions_binary import FUNCTION_DICT_BINARY
from noise_functions_multi import FUNCTION_DICT_MULTI
from mwu import runMWU
import sys
import datetime


def main(arguments):
    parser = argparse.ArgumentParser(description="linear classification experiments argument parser")
    parser.add_argument("-noise_type", help="targeted or untargeted noise", choices=["targeted", "untargeted"],
                        default="untargeted", type=str)
    parser.add_argument("-exp_type", help="binary or multiclass experiments",
                        choices=["binary", "multiclass"], required=True)
    parser.add_argument("-noise_func", help="noise function used for the adversary",
                        choices=["randomAscent", "greedyAscent", "oracle", "gradientDescent", "gradientNonConvex"],
                        required=True)
    parser.add_argument("-iters", help="number of iterations for the MWU", type=int, required=True)
    parser.add_argument("-alpha", help="noise budget", type=float, required=True)
    parser.add_argument("-data_path", help="directory with experiment data + models", type=str, required=True)
    parser.add_argument("-num_classifiers", help="number of classifiers", type=int, required=True)
    args = parser.parse_args(arguments)

    date = datetime.datetime.now()
    exp_name = "{}-{}-{}-{}-{}-{}".format(args.exp_type, args.noise_type, args.noise_func, args.alpha,
                                          date.month, date.day)
    log_file = exp_name + ".log"

    if not os.path.exists(exp_name):
        os.mkdir(exp_name)

    # create log file
    log.basicConfig(format='%(asctime)s: %(message)s', level=log.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

    file_handler = log.FileHandler(exp_name + "/" + log_file)
    log.getLogger().addHandler(file_handler)

    log.debug("{} CLASSIFICATION: \n".format(args.exp_type))
    log.debug("Noise Type {}".format(args.noise_type))
    log.debug("Noise Function {}".format(args.noise_func))
    log.debug("Iters {} ".format(args.iters))
    log.debug("Alpha {}".format(args.alpha))
    log.debug("Data path : {}".format(args.data_path))
    log.debug("Num Classifiers : {}".format(args.num_classifiers))

    X_test = np.load(args.data_path + "/" + "X_test.npy")
    Y_test = np.load(args.data_path + "/" + "Y_test.npy")

    models = []

    for i in xrange(args.num_classifiers):
        weights = np.load(args.data_path + "/" + "weights_{}.npy".format(i))
        bias = np.load(args.data_path + "/" + "bias_{}.npy".format(i))
        if args.exp_type == "binary":
            model = LinearBinaryClassifier(weights, bias)
        else:
            model = LinearOneVsAllClassifier(10, weights, bias)
        log.debug("Model {}, Test Accuracy {}".format(i, model.evaluate(X_test, Y_test)))
        models.append(model)

    X_exp = np.load(args.data_path + "/" + "X_exp.npy")
    Y_exp = np.load(args.data_path + "/" + "Y_exp.npy")

    if args.noise_type == "targeted":
        Target_exp =  np.load(args.data_path + "/" + "Target_exp.npy")
    else:
        Target_exp = False

    log.debug("Num Points {}\n".format(X_exp.shape[0]))

    if args.exp_type == "binary":
        noise_func = FUNCTION_DICT_BINARY[args.noise_func]
    else:
        noise_func = FUNCTION_DICT_MULTI[args.noise_func]

    weights, noise, loss_history, acc_history, action_loss = runMWU(models, args.iters, X_exp, Y_exp, args.alpha,
                                                                    noise_func, exp_name, targeted=Target_exp)

    np.save(exp_name + "/" + "weights.npy", weights)
    np.save(exp_name + "/" + "noise.npy", noise)
    np.save(exp_name + "/" + "loss_history.npy", loss_history)
    np.save(exp_name + "/" + "acc_history.npy", acc_history)
    np.save(exp_name + "/" + "action_loss.npy", action_loss)
    log.debug("Success")

if __name__ == "__main__":
    main(sys.argv[1:])
