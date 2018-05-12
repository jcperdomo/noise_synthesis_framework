## modified from l2_attack.py -- Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified by Juan C. Perdomo 2017

import tensorflow as tf
from tensorflow.contrib.opt import VariableClippingOptimizer
import numpy as np
import logging as log
import time
import sys

MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results, default 1e-2
TARGETED = False         # should we target one specific class? or just be wrong?
CONFIDENCE = 0          # how strong the adversarial example should be


class GradientDescentDL:
    def __init__(self, sess, models, alpha, dataset_params, batch_size=1,
                 confidence=CONFIDENCE, targeted=TARGETED, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS):
        """

        sess: Tensorflow session
        models: list of keras models
        alpha:  noise budget
        dataset_params: tuple of image_size, num_channels, and the number of labels
        box_vals: (min pixel value, max pixel value)
        confidence: scalar, increases margin of adversarial example
        targeted: boolean value
        learning_rate: learning rate for the Adam optimizer
        max_iterations: int
        """
        log.debug("Number of models {} ".format(len(models)))
        image_size, num_channels, num_labels, box_vals = dataset_params  # imagenet parameters 224, 3, 1000 (0, 255)
        self.sess = sess                                                 # mnist parameters 28, 1, 100 (0,1)
        self.alpha = alpha
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.confidence = confidence
        self.batch_size = batch_size
        self.num_models = len(models)
        self.num_labels = num_labels
        self.box_min, self.box_max = box_vals

        shape = (batch_size, image_size, image_size, num_channels)
        
        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size, num_labels)), dtype=tf.float32)
        self.weights = tf.Variable(np.zeros(self.num_models), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size, num_labels))
        self.assign_weights = tf.placeholder(tf.float32, [self.num_models])

        # the resulting image, clipped to keep bounded from boxmin to boxmax
        self.boxmul = (self.box_max - self.box_min) / 2.
        self.boxplus = (self.box_min + self.box_max) / 2.
        # self.newimg = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus
        self.newimg = tf.clip_by_value(modifier + self.timg, self.box_min, self.box_max)

        self.outputs = [model(self.newimg) for model in models]
        
        # compute the probability of the label class versus the maximum other
        reals = []
        others = []
        for i in xrange(self.num_models): # TODO needs to be edited for larger batch sizes
            real = tf.reduce_sum(self.tlab * self.outputs[i], 1)
            other = tf.reduce_max((1 - self.tlab) * self.outputs[i], 1)
            reals.append(real)
            others.append(other)
        self.reals, self.others = reals, others

        loss1list = []

        if self.targeted:
            # if targeted, optimize for making the other class most likely
            for i in xrange(self.num_models):
                loss1list.append(tf.maximum(0.0, self.weights[i] * (others[i] - reals[i] + self.confidence)))
        else:
            # if untargeted, optimize for making this class least likely.
            for i in xrange(self.num_models):
                loss1list.append(tf.maximum(0.0, self.weights[i] * (reals[i] - others[i] + self.confidence)))

        self.loss1list = loss1list

        # sum up the losses
        self.loss1 = tf.add_n(self.loss1list)

        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg - self.timg), [1, 2, 3])
        # self.loss2 = tf.maximum(0.0, self.l2dist - self.alpha ** 2)

        self.loss = self.loss1  # + self.loss2

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        adam = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = VariableClippingOptimizer(adam, {modifier: [1, 2, 3]}, self.alpha)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.weights.assign(self.assign_weights))

        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, imgs, targets, weights):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        for i in range(0, len(imgs), self.batch_size):
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size], weights))
        return np.array(r)

    def attack_batch(self, imgs, labs, weights):
        """
        Run the attack on a batch of images and labels.
        """

        batch_size = len(imgs)
        best_attack = [np.zeros(imgs[0].shape)] * batch_size

        # completely reset adam's internal state.
        self.sess.run(self.init)

        best_loss = [sys.maxint] * batch_size

        # set the variables so that we don't have to send them over again
        self.sess.run(self.setup, {self.assign_timg: imgs,
                                   self.assign_tlab: labs,
                                   self.assign_weights: weights})

        # keras dependency
        from keras import backend as K

        for iteration in xrange(self.max_iterations):
            start_time = time.time()
            # perform the attack
            self.sess.run([self.train], feed_dict={K.learning_phase(): 0})

            nimg, loss1list, l2dist, loss = self.sess.run([self.newimg, self.loss1list, self.l2dist, self.loss],
                                                         feed_dict={K.learning_phase(): 0})

            if iteration == self.max_iterations - 1:
            # if iteration % 1000 == 0:
            #     log.debug("Iteration {}".format(iteration))
                log.debug("Time in Iteration {}".format(time.time() - start_time))
            #     log.debug("Loss1 List {}".format(loss1list))
                log.debug("L2dist {}".format(l2dist))
                # log.debug("Loss {}".format(loss))
                # log.debug("Best Loss {}".format(best_loss))

            # scores = np.array(scores).reshape(self.batch_size, self.num_models, self.num_labels)
            outer_break = False
            for e, im in enumerate(nimg):
                if loss[e] < best_loss[e]: #and loss2[e] == 0:  # we've found a clear improvement for this attack
                    best_loss[e] = loss[e]
                    best_attack[e] = im
                if best_loss[0] == 0.0:  # TODO only works for batch sizes of 1
                    outer_break = True
            if outer_break:
                break

        # return the best solution found
        return np.array(best_attack) - imgs






