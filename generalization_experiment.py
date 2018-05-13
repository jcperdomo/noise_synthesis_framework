from tensorflow.examples.tutorials.mnist import input_data
from sklearn.svm import LinearSVC
import numpy as np
from linear_models import LinearOneVsAllClassifier
from utils import generate_exp_data
from mwu import run_mwu
from noise_functions_multi import grad_desc_nonconvex
import ray
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

mnist_images = np.copy(mnist.train.images)
mnist_labels = np.argmax(mnist.train.labels, axis=1)
mnist_test_images = mnist.test.images
mnist_test_labels = np.argmax(mnist.test.labels, axis=1)

num_mnist_features = 784
num_models = 1000
linear_models = []
zeroed_features_list = []
sparse_training_sets = []

for i in xrange(num_models):
    sparse_mnist_images = np.copy(mnist_images)
    zeroed_features = np.random.choice(range(num_mnist_features), 588, replace=False)
    zeroed_features_list.append(zeroed_features)
    sparse_mnist_images[:, zeroed_features] = 0.0
    sparse_training_sets.append(sparse_mnist_images)


@ray.remote
def train_model(train_set, train_labels):
    model = LinearSVC(loss='hinge')
    model.fit(train_set, train_labels)
    return LinearOneVsAllClassifier(10, model.coef_, model.intercept_)

ray.init()

models = [train_model.remote(train_set, mnist_labels) for train_set in sparse_training_sets]
models = ray.get(models)

exp_folder = 'generalization_experiment'
os.mkdir(exp_folder)
os.mkdir(exp_folder + '/models/')
os.mkdir(exp_folder + '/results')
os.mkdir(exp_folder + '/data/')

for i, model in enumerate(models):
    np.save('{}/models/w_{}.npy'.format(exp_folder, i), model.weights)
    np.save('{}/models/b_{}.npy'.format(exp_folder, i), model.bias)

num_points = 1000
X_exp, Y_exp = generate_exp_data(num_points, mnist_test_images, mnist_test_labels, models)

np.save(exp_folder + '/data/X_exp.npy', X_exp)
np.save(exp_folder + '/data/Y_exp.npy', X_exp)

subset_sizes = [5, 10, 25, 50, 100, 150, 250, 500, 750, 1000]
mwu_iters = 50
alpha = .5
for k in subset_sizes:
    chosen_ixs = np.random.choice(range(num_models), k, replace=False)
    chosen_models = []
    for ix in chosen_ixs:
        chosen_models.append(models[ix])
    weights, noise, loss_history, acc_history, action_loss = run_mwu(chosen_models, mwu_iters, X_exp, Y_exp, alpha,
                                                                     grad_desc_nonconvex)
    np.save(exp_folder + "/results/" + "weights_{}.npy".format(k), weights)
    np.save(exp_folder + "/results/" + "noise_{}.npy".format(k), noise)
    np.save(exp_folder + "/results/" + "loss_history_{}.npy".format(k), loss_history)
    np.save(exp_folder + "/results/" + "acc_history_{}.npy".format(k), acc_history)
    np.save(exp_folder + "/results/" + "action_loss_{}.npy".format(k), action_loss)

print "DONE"
