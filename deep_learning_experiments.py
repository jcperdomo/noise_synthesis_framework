import logging as log
import argparse
from mwu import runMWU
import sys
import datetime
from setup_mnist import *
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


def main(arguments):
    parser = argparse.ArgumentParser(description="deep learning classification experiments argument parser")
    parser.add_argument("-data_set", help="directory with experiment data + models", choices=["mnist", "imagenet"],
                        required=True)
    parser.add_argument("-noise_type", help="targeted or untargeted noise", choices=["targeted", "untargeted"],
                        default="untargeted", type=str)
    parser.add_argument("-mwu_iters", help="number of iterations for the MWU", type=int, required=True)
    parser.add_argument("-alpha", help="noise budget", type=float, required=True)
    parser.add_argument("-opt_iters", help="number of iterations to run optimizer", type=int, required=True)
    parser.add_argument("-learning_rate", help="learning rate for the optimizer", type=float, required=True)
    args = parser.parse_args(arguments)
    
    date = datetime.datetime.now()
    data_path = "multiclass_data_2" if args.data_set == "mnist" else "imagenet_data"
    exp_name = "deepLearning-{}-{}-{}-{}-{}-{}{}".format(args.data_set, args.noise_type, args.alpha, date.month,
                                                         date.day, date.hour, date.minute)
    log_file = exp_name + ".log"

    if not os.path.exists(exp_name):
        os.mkdir(exp_name)

    # create log file
    log.basicConfig(format='%(asctime)s: %(message)s', level=log.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

    file_handler = log.FileHandler(exp_name + "/" + log_file)
    log.getLogger().addHandler(file_handler)

    log.debug("Noise Type {}".format(args.noise_type))
    log.debug("MWU Iters {} ".format(args.mwu_iters))
    log.debug("Alpha {}".format(args.alpha))
    log.debug("Learning Rate {}".format(args.learning_rate))
    log.debug("Optimization Iters {}".format(args.opt_iters))
    log.debug("Data path : {}".format(data_path))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        log.debug("\nbeginning to load models...")

        # setup models
        if args.data_set == "mnist":
            model_dir = "deep_networks"
            models = [conv_net(False, 2, 200, model_dir + "/conv0"), conv_net(True, 2, 200, model_dir + "/conv1"),
                      conv_net(True, 4, 64, model_dir + "/conv2"), multilayer(4, 128, model_dir + "/mlp0"),
                      multilayer(2, 256, model_dir + "/mlp1")]
            for model in models:
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            input_tensor = Input(shape=(224, 224, 3))
            tf_inputs = Lambda(lambda x: preprocess_input(x, mode='tf'))(input_tensor)
            caffe_inputs = Lambda(lambda x: preprocess_input(x, mode='caffe'))(input_tensor)

            base_inception = InceptionV3(input_tensor=input_tensor, weights="imagenet", include_top=True)
            inception = Model(inputs=input_tensor, outputs=base_inception(tf_inputs))

            base_resnet = ResNet50(input_tensor=input_tensor, weights="imagenet", include_top=True)
            resnet = Model(inputs=input_tensor, outputs=base_resnet(caffe_inputs))

            base_inceptionresnet = InceptionResNetV2(input_tensor=input_tensor, weights="imagenet", include_top=True)
            inceptionresnet = Model(inputs=input_tensor, outputs=base_inceptionresnet(tf_inputs))

            base_vgg = VGG19(input_tensor=input_tensor, weights="imagenet", include_top=True)
            vgg = Model(inputs=input_tensor, outputs=base_vgg(caffe_inputs))

            models = [vgg, inceptionresnet, resnet, inception]

            for model in models:
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        log.debug("finished loading models!\n")

        X_exp = np.load(data_path + "/" + "X_exp.npy")
        Y_exp = np.load(data_path + "/" + "Y_exp.npy")
        Target_exp = np.load(data_path + "/" + "Target_exp.npy")

        # fit dta parameters
        if args.data_set == "mnist":
            X_exp = X_exp.reshape(-1, 28, 28, 1)
            Y_exp = np.array([(np.arange(10) == l).astype(np.float32) for l in Y_exp])
            Target_exp = np.array([(np.arange(10) == l).astype(np.float32) for l in Target_exp])
            data_dims = [28, 1, 10, (-.5, .5)]
        else:  # "imagenet"
            data_dims = [224, 3, 1000, (0.0, 255.0)]
            X_exp = X_exp[:50]
            Y_exp = Y_exp[:50]
            Target_exp = Target_exp[:50]

        log.debug("Num Points {}".format(X_exp.shape[0]))
        target_bool = args.noise_type == "targeted"

        # initialize the attack object
        attack_obj = GradientDescentDL(sess, models, args.alpha, data_dims, box_vals, targeted=target_bool,
                                       batch_size=1, max_iterations=args.opt_iters, learning_rate=args.learning_rate,
                                       confidence=0)

        log.debug("starting attack!")
        noise_func = partial(gradientDescentFunc, attack=attack_obj)
        targeted = Target_exp if target_bool else False
        weights, noise, loss_history, acc_history, action_loss = runMWU(models, args.mwu_iters, X_exp, Y_exp, args.alpha,
                                                                        noise_func, exp_name, targeted=targeted,
                                                                        dl=True)

        np.save(exp_name + "/" + "weights.npy", weights)
        np.save(exp_name + "/" + "noise.npy", noise)
        np.save(exp_name + "/" + "loss_history.npy", loss_history)
        np.save(exp_name + "/" + "acc_history.npy", acc_history)
        np.save(exp_name + "/" + "action_loss.npy", action_loss)

        log.debug("Success")

if __name__ == "__main__":
    main(sys.argv[1:])
