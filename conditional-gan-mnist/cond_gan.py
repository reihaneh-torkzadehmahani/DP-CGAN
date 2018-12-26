"""
Reference for Vanilla GAN code:
https://github.com/wiseodd/generative-models/blob/master/GAN/

Reference for Moment accountant code:
https://github.com/tensorflow/models/tree/master/research/differential_privacy
"""

# base directory for the project
baseDir = "/content/gdrive/Team Drives/PrivacyGenomics/"
max_iteration_count = 25200
batch_size = 600

# Import the requiered python packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys


from mlxtend.data import loadlocal_mnist
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import label_binarize



def compute_fpr_tpr_roc(Y_test, Y_score):
    n_classes = Y_score.shape[1]
    false_positive_rate = dict()
    true_positive_rate = dict()
    roc_auc = dict()
    for class_cntr in range(n_classes):
        false_positive_rate[class_cntr], true_positive_rate[class_cntr], _ = roc_curve(Y_test[:, class_cntr],
                                                                                       Y_score[:, class_cntr])
        roc_auc[class_cntr] = auc(false_positive_rate[class_cntr], true_positive_rate[class_cntr])

    # Compute micro-average ROC curve and ROC area
    false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

    return false_positive_rate, true_positive_rate, roc_auc


def classify(X_train, Y_train, X_test, classiferName, random_state_value=0):
    if classiferName == "svm":
        classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state_value))
    elif classiferName == "dt":
        classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=random_state_value))
    elif classiferName == "lr":
        classifier = OneVsRestClassifier(
            LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=random_state_value))
    elif classiferName == "rf":
        classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state_value))
    elif classiferName == "gnb":
        classifier = OneVsRestClassifier(GaussianNB())
    elif classiferName == "bnb":
        classifier = OneVsRestClassifier(BernoulliNB(alpha=.01))
    elif classiferName == "ab":
        classifier = OneVsRestClassifier(AdaBoostClassifier(random_state=random_state_value))
    elif classiferName == "mlp":
        classifier = OneVsRestClassifier(MLPClassifier(random_state=random_state_value, alpha=1))
    else:
        print("Classifier not in the list!")
        exit()

    Y_score = classifier.fit(X_train, Y_train).predict_proba(X_test)
    return Y_score

# Import required Differential Privacy packages
sys.path.append(baseDir);


def xavier_init(size):
    """ Xavier Function to keep the scale of the gradients roughly the same
        in all the layers.
    """
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Initializations for a two-layer discriminator network
mnist = input_data.read_data_sets(baseDir + "/conditional-dpgan-mnist/mnist_dataset", one_hot=True)
h_dim = 128
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
Z_dim = 100
X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_b1, D_b2]

# Initializations for a two-layer genrator network
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))
theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    """ Function to generate uniform prior for G(z)
    """
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z, y):
    """ Function to build the generator network
    """
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x, y):
    """ Function to build the discriminator network
    """
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    """ Function to plot the generated images
    """
    fig = plt.figure(figsize=(4, 5))
    gs = gridspec.GridSpec(4, 5)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        # plt.show()
    return fig


def del_all_flags(FLAGS):
    """ Function to delete all flags before declare
    """
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


# Delete all Flags
del_all_flags(tf.flags.FLAGS)

# Set accountant type to GaussianMomentsAccountant
#NUM_TRAINING_IMAGES = 60000


# Instantiate the Generator Network
G_sample = generator(Z, y)

# Instantiate the Discriminator Network
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

# Discriminator loss for real data
D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits( \
        logits=D_logit_real, \
        labels=tf.ones_like(D_logit_real)), \
    [0])
# Discriminator loss for fake data
D_loss_fake = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits( \
        logits=D_logit_fake, \
        labels=tf.zeros_like(D_logit_fake)), [0])
D_loss = D_loss_real+D_loss_fake

# Generator loss
G_loss = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits( \
        logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)) \
    , [0])

# Generator optimizer
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


# Discriminator Optimizer
D_solver = tf.train.AdamOptimizer().minimize(D_loss_real + D_loss_fake, var_list=theta_D)
# ------------------------------------------------------------------------------

# Set output directory
resultDir = baseDir + "conditional-gan-mnist/results/"
if not os.path.exists(resultDir):
    os.makedirs(resultDir)

resultPath = resultDir + "bs_{}".format(batch_size)


if not os.path.exists(resultPath):
    os.makedirs(resultPath)

# Delete all Flags
del_all_flags(tf.flags.FLAGS)


# Main Session
with tf.train.SingularMonitoredSession() as sess:
    step = 0

    # Main loop
    while (step < max_iteration_count):

        X_mb, y_mb = mnist.train.next_batch(batch_size, shuffle=True)
        Z_sample = sample_Z(batch_size, Z_dim)

        # Update the discriminator network
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y: y_mb})
        # Update the generator network
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y: y_mb})

        if step % 100 == 0:
            print("Step: " + str(step))
        # Save the generated images every 100 steps
        if step == (max_iteration_count - 1):
            n_class = np.zeros(10)

            n_class[0] = 5923
            n_class[1] = 6742
            n_class[2] = 5958
            n_class[3] = 6131
            n_class[4] = 5842
            n_class[5] = 5421
            n_class[6] = 5918
            n_class[7] = 6265
            n_class[8] = 5851
            n_class[9] = 5949

            n_image = int(sum(n_class))
            image_lables = np.zeros(shape=[n_image, len(n_class)])

            image_cntr = 0
            for class_cntr in np.arange(len(n_class)):
                for cntr in np.arange(n_class[class_cntr]):
                    image_lables[image_cntr, class_cntr] = 1
                    image_cntr += 1

            Z_sample = sample_Z(n_image, Z_dim)

            images = sess.run(G_sample, feed_dict={Z: Z_sample, y: image_lables})

            X_test, Y_test = loadlocal_mnist(
                images_path=baseDir + 'mnist_dataset/t10k-images.idx3-ubyte',
                labels_path=baseDir + 'mnist_dataset/t10k-labels.idx1-ubyte')

            Y_test = [int(y) for y in Y_test]

            # X = X.reshape((X.shape[0], -1))

            print("Binarizing the labels ...")
            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            Y_test = label_binarize(Y_test, classes=classes)

            print("\n################# Logistic Regression #######################")

            print("  Classifying ...")
            Y_score = classify(images, image_lables, X_test, "lr", random_state_value=30)

            print("  Computing ROC ...")
            false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(Y_test, Y_score)
            print("  AUROC: " + str(roc_auc["micro"]))

            print("\n################# Random Forest #######################")

            print("  Classifying ...")
            Y_score = classify(images, image_lables, X_test, "rf", random_state_value=30)

            print("  Computing ROC ...")
            false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(Y_test, Y_score)
            print("  AUROC: " + str(roc_auc["micro"]))

            print("\n################# Gaussian Naive Bayes #######################")

            print("  Classifying ...")
            Y_score = classify(images, image_lables, X_test, "gnb", random_state_value=30)

            print("  Computing ROC ...")
            false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(Y_test, Y_score)
            print("  AUROC: " + str(roc_auc["micro"]))

            print("\n################# Decision Tree #######################")

            print("  Classifying ...")
            Y_score = classify(images, image_lables, X_test, "dt", random_state_value=30)

            print("  Computing ROC ...")
            false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(Y_test, Y_score)
            print("  AUROC: " + str(roc_auc["micro"]))

            print("\n################# Multi-layer Perceptron #######################")

            print("  Classifying ...")
            Y_score = classify(images, image_lables, X_test, "mlp", random_state_value=30)

            print("  Computing ROC ...")
            false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(Y_test, Y_score)
            print("  AUROC: " + str(roc_auc["micro"]))

        step = step + 1


