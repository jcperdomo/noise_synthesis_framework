import logging as log
import argparse
import os
import numpy as np
from linear_models import LinearBinaryClassifier, LinearOneVsAllClassifier
from noise_functions_binary import FUNCTION_DICT_BINARY
from noise_functions_multi import FUNCTION_DICT_MULTI
from mwu import run_mwu
import sys
import datetime


def main(arguments):
    parser = argparse.ArgumentParser(description="linear classification experiments argument parser")
    parser.add_argument("-exp_type", help="binary or multiclass experiments",
                        choices=["binary", "multi"], required=True)
    parser.add_argument("-noise_type", help="targeted or untargeted noise", choices=["targeted", "untargeted"],
                        default="untargeted", type=str)
    parser.add_argument("-noise_func", help="noise function used for the adversary",
                        choices=["oracle", "grad_desc_convex", "grad_desc_nonconvex", 'grad_desc'],
                        required=True)
    parser.add_argument("-num_classifiers", help='number of classifiers', type=int, required=True)
    parser.add_argument("-iters", help="number of iterations for the MWU", type=int, required=True)
    parser.add_argument("-alpha", help="noise budget", type=float, required=True)
    parser.add_argument("-log_level", help='level of info for the logger', choices=['INFO', 'DEBUG'], required=True)
    parser.add_argument("-model_path", help="directory with model data ", type=str, required=True)
    parser.add_argument("-data_path", help="directory with experiment data ", type=str, required=True)
    parser.add_argument("-purpose", help='short string (1 word) to describe purpose of experiment', type=str,
                        required=True)
    args = parser.parse_args(arguments)

    date = datetime.datetime.now()
    exp_name = "linear_{}_{}_{}_{}_{}_{}_{}".format(args.purpose, args.exp_type, args.noise_type, args.noise_func,
                                                    args.alpha, date.month, date.day)

    exp_dir = 'experiment_results/' + exp_name
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    # create log file
    log_file = exp_name + ".log"
    log_level = log.DEBUG if args.log_level == 'DEBUG' else log.INFO
    log.basicConfig(format='%(asctime)s: %(message)s', level=log_level, datefmt='%m/%d/%Y %I:%M:%S %p')
    file_handler = log.FileHandler(exp_dir + "/" + log_file)
    log.getLogger().addHandler(file_handler)

    log.info("Experiment Type {}".format(args.exp_type))
    log.info("Noise Type {}".format(args.noise_type))
    log.info("Noise Function {}".format(args.noise_func))
    log.info("Iters {}".format(args.iters))
    log.info("Alpha {}".format(args.alpha))
    log.info("Data path {}".format(args.data_path))
    log.info("Model path {}".format(args.model_path))
    log.info("Num Classifiers {}".format(args.num_classifiers))

    linear_models = []
    for i in xrange(args.num_classifiers):
        weights = np.load(args.model_path + "/" + "w_{}.npy".format(i))
        bias = np.load(args.model_path + "/" + "b_{}.npy".format(i))
        if args.exp_type == "binary":
            model = LinearBinaryClassifier(weights, bias)
        else:
            model = LinearOneVsAllClassifier(10, weights, bias)
        linear_models.append(model)

    X_exp = np.load(args.data_path + "/" + "X_exp.npy")
    Y_exp = np.load(args.data_path + "/" + "Y_exp.npy")

    if args.noise_type == "targeted":
        Targets_exp = np.load(args.data_path + "/" + "Targets_exp.npy")
    else:
        Targets_exp = False

    log.info("Num Points {}\n".format(X_exp.shape[0]))

    if args.exp_type == "binary":
        noise_func = FUNCTION_DICT_BINARY[args.noise_func]
    else:
        noise_func = FUNCTION_DICT_MULTI[args.noise_func]

    weights, noise, loss_history, acc_history, action_loss = run_mwu(linear_models, args.iters, X_exp, Y_exp,
                                                                     args.alpha, noise_func, targeted=Targets_exp)

    np.save(exp_dir + "/" + "weights.npy", weights)
    np.save(exp_dir + "/" + "noise.npy", noise)
    np.save(exp_dir + "/" + "loss_history.npy", loss_history)
    np.save(exp_dir + "/" + "acc_history.npy", acc_history)
    np.save(exp_dir + "/" + "action_loss.npy", action_loss)
    log.info("Success")

if __name__ == "__main__":
    main(sys.argv[1:])
