import math
import pandas

from mlxtend.data import loadlocal_mnist
from mlxtend.preprocessing import one_hot
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

# Import the requiered python packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

import pyreadr


# Import required Differential Privacy packages
### To run locally
##sys.path.append('models/research')
##sys.path.append('models/research/slim')
##baseDir = ""
##sys.path.append(baseDir);

### TO tun in GoogleColab
baseDir = "/content/gdrive/Team Drives/PrivacyGenomics/"
sys.path.append(baseDir);

from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from differential_privacy.dp_sgd.dp_optimizer import sanitizer
from differential_privacy.dp_sgd.dp_optimizer import utils
from differential_privacy.privacy_accountant.tf import accountant





sigmaList = [2]
batchSizeList = [600]
clippingValueList = [2]
epsilon = 8
delta = 1e-5

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

def xavier_init(size):
    """ Xavier Function to keep the scale of the gradients roughly the same
        in all the layers.
    """
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_Z(m, n):
    """ Function to generate uniform prior for G(z)
    """
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z, y, theta_G):
    G_W1 = theta_G[0]
    G_W2 = theta_G[1]
    G_b1 = theta_G[2]
    G_b2 = theta_G[3]

    """ Function to build the generator network
    """
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x, y, theta_D):
    """ Function to build the discriminator network
    """
    D_W1 = theta_D[0]
    D_W2 = theta_D[1]
    D_b1 = theta_D[2]
    D_b2 = theta_D[3]

    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    """ Function to plot the generated images
    """
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(10, 1)
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



def runTensorFlow(sigma, clippingValue, batchSize, epsilon, delta):

    credit_card_df = pandas.read_csv('/home/reihaneh/Downloads/creditcard.csv', header=0)  # also works for Rds

    # done! let's see what we got
    # result is a dictionary where keys are the name of objects and the values python
    # objects
    #print(credit_card_dataset.keys())  # let's check what objects we got

    #creditcardData = credit_card_dataset.values  # extract the pandas data frame for object df1

    #dataset_size = creditcardData.shape[0]

    dataset = credit_card_df.values
    dataset_size = dataset.shape[0]

    np.take(dataset, np.random.permutation(dataset_size), axis=0, out=dataset)


    dataset_X = dataset[:,1:29]

    dataset_Y = ((dataset[:,30]))
    dataset_Y = [int(x) for x in dataset_Y]
    dataset_Y = one_hot(dataset_Y)

    h_dim = 128
    Z_dim = 2

    # Initializations for a two-layer discriminator network
    ##mnist = input_data.read_data_sets(baseDir + "conditional-gan-dp-ours-mnist/mnist_dataset", one_hot=True)

    ##X_dim = mnist.train.images.shape[1]
    X_dim = dataset_X.shape[1]

    ##y_dim = mnist.train.labels.shape[1]
    y_dim = 2

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

    # Delete all Flags
    del_all_flags(tf.flags.FLAGS)

    # Set training parameters
    tf.flags.DEFINE_string('f', '', 'kernel')
    tf.flags.DEFINE_float("lr", 0.05, "start learning rate")
    tf.flags.DEFINE_float("end_lr", 0.05, "end learning rate")
    tf.flags.DEFINE_float("lr_saturate_epochs", 0,
                          "learning rate saturate epochs; set to 0 for a constant"
                          "learning rate of --lr.")
    tf.flags.DEFINE_integer("batch_size", batchSize, "The training batch size.")
    tf.flags.DEFINE_integer("batches_per_lot", 1, "Number of batches per lot.")
    tf.flags.DEFINE_integer("num_training_steps", 100000, "The number of training"
                                                          "steps. This counts number of lots.")

    # Flags that control privacy spending during training
    tf.flags.DEFINE_float("target_delta", delta, "Maximum delta for"
                                                "--terminate_based_on_privacy.")
    tf.flags.DEFINE_float("sigma", sigma, "Noise sigma, used only if accountant_type"
                                      "is Moments")
    tf.flags.DEFINE_string("target_eps", str(epsilon),
                           "Log the privacy loss for the target epsilon's. Only"
                           "used when accountant_type is Moments.")
    tf.flags.DEFINE_float("default_gradient_l2norm_bound", clippingValue, "norm clipping")

    FLAGS = tf.flags.FLAGS

    # Set accountant type to GaussianMomentsAccountant
    train_size = int(math.floor(int(dataset_size) * 0.9))


    priv_accountant = accountant.GaussianMomentsAccountant(train_size)

    # Sanitizer
    batch_size = FLAGS.batch_size
    clipping_value = FLAGS.default_gradient_l2norm_bound
    gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(priv_accountant,
                                                              [clipping_value / batch_size, True])

    # Instantiate the Generator Network
    G_sample = generator(Z, y, theta_G)

    # Instantiate the Discriminator Network
    D_real, D_logit_real = discriminator(X, y, theta_D)
    D_fake, D_logit_fake = discriminator(G_sample, y, theta_D)

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

    # Generator loss
    G_loss = tf.reduce_mean( \
        tf.nn.sigmoid_cross_entropy_with_logits( \
            logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)) \
        , [0])

    # Generator optimizer
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    # Discriminator Optimizer
    # ------------------------------------------------------------------------------
    """
    minimize_ours :
            Our method (Clipping the gradients of loss on real data and making
            them noisy + Clipping the gradients of loss on fake data) is
            implemented in this function .
            It can be found in the following directory:
            /content/gdrive/Team Drives/PrivacyGenomics/our_dp_gan/
            differential_privacy/dp_sgd/dp_optimizer/dp_optimizer.py'
    """
    lr = tf.placeholder(tf.float32)
    sigma = FLAGS.sigma
    D_solver = dp_optimizer.DPGradientDescentOptimizer( \
        lr, [None, None], \
        gaussian_sanitizer, \
        sigma=sigma, \
        batches_per_lot= \
            FLAGS.batches_per_lot). \
        minimize_ours( \
        D_loss_real, \
        D_loss_fake, \
        var_list=theta_D)
    # ------------------------------------------------------------------------------

    target_eps = [float(s) for s in FLAGS.target_eps.split(",")]
    max_target_eps = max(target_eps)

    # Main Session
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        lot_size = FLAGS.batches_per_lot * batch_size
        lots_per_epoch = train_size / lot_size
        step = 0

        # Is true when the spent privacy budget exceeds the target budget
        should_terminate = False

        # Main loop
        while (step < FLAGS.num_training_steps and should_terminate == False):

            epoch = step / lots_per_epoch
            curr_lr = utils.VaryRate(FLAGS.lr, FLAGS.end_lr, \
                                     FLAGS.lr_saturate_epochs, epoch)



            batch_index = 0
            for _ in range(FLAGS.batches_per_lot):

                ##X_mb, y_mb = mnist.train.next_batch(batch_size, shuffle=True)
                if (batch_index + batch_size - 1 > train_size):
                    break

                X_mb = dataset_X[batch_index:batch_size]
                y_mb = dataset_Y[batch_index:batch_size]
                #y_mb = label_binarize(y_mb, [0,1])
                #y_mb = one_hot(y_mb , num_labels=2)

                Z_sample = sample_Z(batch_size, Z_dim)

                # Update the discriminator network
                _, D_loss_real_curr, D_loss_fake_curr= sess.run([D_solver, D_loss_real, D_loss_fake], \
                                             feed_dict={X: X_mb, \
                                                        Z: Z_sample, \
                                                        y: y_mb, \
                                                        lr: curr_lr})
                # Update the generator network
                _, G_loss_curr = sess.run([G_solver, G_loss],
                                          feed_dict={Z: Z_sample, y: y_mb})
                batch_index += batch_size

            # Flag to terminate based on target privacy budget
            terminate_spent_eps_delta = priv_accountant.get_privacy_spent(sess, \
                                                                          target_eps=[max_target_eps])[0]

            # For the Moments accountant, we should always have \
            # spent_eps == max_target_eps.
            if (terminate_spent_eps_delta.spent_delta > FLAGS.target_delta or \
                            terminate_spent_eps_delta.spent_eps > max_target_eps):
                spent_eps_deltas = priv_accountant.get_privacy_spent( \
                    sess, target_eps=target_eps)
                print("TERMINATE!!!!")
                print("Termination Step : " + str(step))
                should_terminate = True

                return

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


for sigma in sigmaList:
    for clippingValue in clippingValueList:
        for batchSize in batchSizeList:
            print ("Running TensorFlow with Sigma=%f, Clipping=%d, batchSize=%d\n" % (sigma, clippingValue, batchSize))
            runTensorFlow(sigma, float(clippingValue), batchSize, epsilon, delta)
