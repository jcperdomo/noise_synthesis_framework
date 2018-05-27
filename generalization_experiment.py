from tensorflow.examples.tutorials.mnist import input_data
from sklearn.svm import LinearSVC
import numpy as np
from linear_models import LinearOneVsAllClassifier
from utils import generate_exp_data, subset_multiclass_data
from mwu import run_mwu
from noise_functions_multi import grad_desc_convex
import ray
import time
import os

ray.init()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

mnist_train_images = np.copy(mnist.train.images)
mnist_train_labels = np.argmax(mnist.train.labels, axis=1)
mnist_test_images = mnist.test.images
mnist_test_labels = np.argmax(mnist.test.labels, axis=1)

label_dict = {0: 0, 1: 1, 2: 2, 3: 3}
multi_train_images, multi_train_labels = subset_multiclass_data(mnist_train_images, mnist_train_labels, label_dict)
multi_test_images, multi_test_labels = subset_multiclass_data(mnist_test_images, mnist_test_labels, label_dict)

num_models = 50
num_classes = 4
mnist_num_dim = 28 * 28
sparse_features_perc = .75
models = []
zeroed_features_list = []
for i in xrange(num_models):
    sparse_multi_images = np.copy(multi_train_images)
    zeroed_features = np.random.randint(0, mnist_num_dim, int(sparse_features_perc * mnist_num_dim))
    zeroed_features_list.append(zeroed_features)
    sparse_multi_images[:, zeroed_features] = 0.0
    model = LinearSVC(loss='hinge')
    model.fit(sparse_multi_images, multi_train_labels)
    model = LinearOneVsAllClassifier(num_classes, model.coef_, model.intercept_)
    models.append(model)

exp_folder = 'generalization_experiment'
os.mkdir(exp_folder)
os.mkdir(exp_folder + '/models/')
os.mkdir(exp_folder + '/results')
os.mkdir(exp_folder + '/data/')

for i, model in enumerate(models):
    np.save('{}/models/w_{}.npy'.format(exp_folder, i), model.weights)
    np.save('{}/models/b_{}.npy'.format(exp_folder, i), model.bias)

models = []
for i in xrange(num_models):
    w = np.load('{}/models/w_{}.npy'.format(exp_folder, i))
    b = np.load('{}/models/b_{}.npy'.format(exp_folder, i))
    models.append(LinearOneVsAllClassifier(10, w, b))

print("Done Saving models")

num_points = 1000
X_exp, Y_exp = generate_exp_data(num_points, mnist_test_images, mnist_test_labels, models)


print("number of points {}".format(X_exp.shape))

np.save(exp_folder + '/data/X_exp.npy', X_exp)
np.save(exp_folder + '/data/Y_exp.npy', Y_exp)

# X_exp = np.load(exp_folder + '/data/X_exp.npy')
# Y_exp = np.load(exp_folder + '/data/Y_exp.npy')

subset_sizes = [1, 5, 10, 20, 30, 40, 50]
mwu_iters = 10
alpha = .8

for k in subset_sizes:
    print("Iteration {}".format(k))
    start = time.time()
    chosen_ixs = np.random.choice(range(num_models), k, replace=False)
    chosen_models = []
    for ix in chosen_ixs:
        chosen_models.append(models[ix])

    weights, noise, loss_history, acc_history, action_loss = run_mwu(chosen_models, mwu_iters, X_exp, Y_exp, alpha,
                                                                     grad_desc_convex)

    np.save(exp_folder + "/results/" + "weights_{}.npy".format(k), weights)
    np.save(exp_folder + "/results/" + "noise_{}.npy".format(k), noise)
    np.save(exp_folder + "/results/" + "loss_history_{}.npy".format(k), loss_history)
    np.save(exp_folder + "/results/" + "acc_history_{}.npy".format(k), acc_history)
    np.save(exp_folder + "/results/" + "action_loss_{}.npy".format(k), action_loss)

    time_in_iter = time.time() - start
    print("Time in iter {}".format(time_in_iter))

print("DONE")
