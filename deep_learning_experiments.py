import logging as log
import argparse
from mwu import run_mwu
import sys
import datetime
from functools import partial
import numpy as np
import tensorflow as tf
import os
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers.core import Lambda
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from noise_functions_dl import GradientDescentDL, gradientDescentFunc
from mnist_dl_models import load_model


def main(arguments):
    parser = argparse.ArgumentParser(description="deep learning classification experiments argument parser")
    parser.add_argument("-exp_type", help="mnist or imagenet experiments", choices=["mnist", "imagenet"], required=True)
    parser.add_argument("-noise_type", help="targeted or untargeted noise", choices=["targeted", "untargeted"],
                        type=str, required=True)
    parser.add_argument("-num_classifiers", help='number of classifiers', type=int, required=True)
    parser.add_argument("-data_path", help="directory with experiment data", type=str, required=True)
    parser.add_argument("-model_path", help="directory with model weights", type=str, required=True)
    parser.add_argument("-mwu_iters", help="number of iterations for the MWU", type=int, required=True)
    parser.add_argument("-alpha", help="noise budget", type=float, required=True)
    parser.add_argument("-opt_iters", help="number of iterations to run optimizer", type=int, required=True)
    parser.add_argument("-learning_rate", help="learning rate for the optimizer", type=float, required=True)
    parser.add_argument("-log_level", help='level of info for the logger', choices=['INFO', 'DEBUG'], required=True)
    parser.add_argument("-purpose", help='short string (1 word) to describe purpose of experiment', type=str,
                        required=True)
    args = parser.parse_args(arguments)
    
    date = datetime.datetime.now()
    exp_name = "deepLearning_{}_{}_{}_{}_{}_{}".format(args.exp_type, args.purpose, args.noise_type, args.alpha,
                                                       date.month, date.day)

    exp_dir = 'experiment_results/' + exp_name
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    log_file = exp_name + ".log"
    log_level = log.DEBUG if args.log_level == 'DEBUG' else log.INFO
    log.basicConfig(format='%(asctime)s: %(message)s', level=log_level, datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename=exp_dir + "/" + log_file, filemode='w')
    # file_handler = log.FileHandler(exp_dir + "/" + log_file)
    # log.getLogger().addHandler(file_handler)

    log.info("Experiment Type {}".format(args.exp_type))
    log.info("Noise Type {}".format(args.noise_type))
    log.info("Num Classifiers {}".format(args.num_classifiers))
    log.info("MWU Iters {} ".format(args.mwu_iters))
    log.info("Alpha {}".format(args.alpha))
    log.info("Learning Rate {}".format(args.learning_rate))
    log.info("Optimization Iters {}".format(args.opt_iters))
    log.info("Data path : {}".format(args.data_path))
    log.info("Model path {}".format(args.model_path))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        log.debug("\nbeginning to load models...")

        # setup models
        if args.exp_type == "mnist":
            models = [load_model(i, '{}/model_{}_weights.h5'.format(args.model_path, i))
                      for i in range(args.num_classifiers)]
            dataset_params = [28, 1, 10, (0.0, 1.0)]
        else:
            # input_tensor = Input(shape=(224, 224, 3))
            # tf_inputs = Lambda(lambda x: preprocess_input(x, mode='tf'))(input_tensor)
            # caffe_inputs = Lambda(lambda x: preprocess_input(x, mode='caffe'))(input_tensor)
            #
            # base_inception = InceptionV3(input_tensor=input_tensor, weights="imagenet", include_top=True)
            # inception = Model(inputs=input_tensor, outputs=base_inception(tf_inputs))
            #
            # base_resnet = ResNet50(input_tensor=input_tensor, weights="imagenet", include_top=True)
            # resnet = Model(inputs=input_tensor, outputs=base_resnet(caffe_inputs))
            #
            # base_inceptionresnet = InceptionResNetV2(input_tensor=input_tensor, weights="imagenet", include_top=True)
            # inceptionresnet = Model(inputs=input_tensor, outputs=base_inceptionresnet(tf_inputs))
            #
            # base_vgg = VGG19(input_tensor=input_tensor, weights="imagenet", include_top=True)
            # vgg = Model(inputs=input_tensor, outputs=base_vgg(caffe_inputs))
            #
            # models = [vgg, inceptionresnet, resnet, inception]
            #
            # for model in models:
            #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            #
            dataset_params = [224, 3, 1000, (0.0, 255.0)]

        log.debug("finished loading models!\n")

        X_exp = np.load(args.data_path + "/" + "X_exp.npy")
        Y_exp = np.load(args.data_path + "/" + "Y_exp.npy")
        # Target_exp = np.load(data_path + "/" + "Target_exp.npy")

        log.info("Num Points {}".format(X_exp.shape[0]))
        target_bool = args.noise_type == "targeted"

        # initialize the attack object
        attack_obj = GradientDescentDL(sess, models, args.alpha, dataset_params, targeted=target_bool,
                                       batch_size=1, max_iterations=args.opt_iters, learning_rate=args.learning_rate,
                                       confidence=0)

        noise_func = partial(gradientDescentFunc, attack=attack_obj)
        # targeted = Target_exp if target_bool else False
        targeted = False
        weights, noise, loss_history, acc_history, action_loss = run_mwu(models, args.mwu_iters, X_exp, Y_exp,
                                                                         args.alpha, noise_func, targeted=targeted,
                                                                         dl=True, use_ray=False)

        np.save(exp_dir + "/" + "weights.npy", weights)
        np.save(exp_dir + "/" + "noise.npy", noise)
        np.save(exp_dir + "/" + "loss_history.npy", loss_history)
        np.save(exp_dir + "/" + "acc_history.npy", acc_history)
        np.save(exp_dir + "/" + "action_loss.npy", action_loss)
        log.info("Success")

if __name__ == "__main__":
    main(sys.argv[1:])
