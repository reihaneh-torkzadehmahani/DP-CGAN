# -*- coding: utf-8 -*-
from __future__ import division

import time

from differential_privacy.optimizer import dp_optimizer

from gan.ops import *
from gan.utils import *

from differential_privacy.analysis.rdp_accountant import compute_rdp
from differential_privacy.analysis.rdp_accountant import get_privacy_spent

from mlxtend.data import loadlocal_mnist
from sklearn.preprocessing import label_binarize

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


class OUR_DP_CGAN(object):
    model_name = "OUR_DP_CGAN"  # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, epsilon, delta, sigma, clip_value, lr, dataset_name, base_dir,
                 checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.base_dir = base_dir
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = sigma
        self.l2_norm_clip = clip_value
        self.lr = lr

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim  # dimension of noise-vector
            self.y_dim = 10  # dimension of condition-vector (label)
            self.c_dim = 1

            # train
            self.learningRateD = self.lr
            self.learningRateG = self.learningRateD * 5
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name, self.base_dir + "data/")

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size


        elif dataset_name == 'cifar10' :
            # parameters
            self.input_height = 32
            self.input_width = 32
            self.output_height = 32
            self.output_width = 32

            self.z_dim = 100  # dimension of noise-vector
            self.y_dim = 10  # dimension of condition-vector (label)
            self.c_dim = 3  # color dimension

            # train
            # self.learning_rate = 0.0002 # 1e-3, 1e-4
            self.learningRateD = 1e-3
            self.learningRateG = 1e-4
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load cifar10
            self.data_X, self.data_y = load_cifar10()

            self.num_batches = len(self.data_X) // self.batch_size

        else:
            raise NotImplementedError

    def discriminator(self, x, y, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):

            # merge image and label
            if (self.dataset_name == "mnist"):
                y = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(x, y)

                net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
                net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
                net = tf.reshape(net, [self.batch_size, -1])
                net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
                out_logit = linear(net, 1, scope='d_fc4')
                out = tf.nn.sigmoid(out_logit)

            elif (self.dataset_name == "cifar10"):

                print("D:", x.get_shape())  # 32, 32, 3 = 3072
                net = lrelu(conv2d(x, 64, 5, 5, 2, 2, name='d_conv1' + '_' + self.dataset_name))
                print("D:", net.get_shape())
                net = lrelu(bn(conv2d(net, 128, 5, 5, 2, 2, name='d_conv2' + '_' + self.dataset_name),
                               is_training=is_training, scope='d_bn2'))
                print("D:", net.get_shape())
                net = lrelu(bn(conv2d(net, 256, 5, 5, 2, 2, name='d_conv3' + '_' + self.dataset_name),
                               is_training=is_training, scope='d_bn3'))
                print("D:", net.get_shape())
                net = lrelu(bn(conv2d(net, 512, 5, 5, 2, 2, name='d_conv4' + '_' + self.dataset_name),
                               is_training=is_training, scope='d_bn4'))
                print("D:", net.get_shape())
                net = tf.reshape(net, [self.batch_size, -1])
                print("D:", net.get_shape())
                out_logit = linear(net, 1, scope='d_fc5' + '_' + self.dataset_name)
                print("D:", net.get_shape())
                out = tf.nn.sigmoid(out_logit)
                print("D:", out.get_shape())
                print("------------------------")

            return out, out_logit, net

    def generator(self, z, y, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):

            if (self.dataset_name == "mnist"):
                # merge noise and label
                z = concat([z, y], 1)

                net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
                net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
                net = tf.reshape(net, [self.batch_size, 7, 7, 128])
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
                       scope='g_bn3'))

                out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

            elif (self.dataset_name == "cifar10"):
                h_size = 32
                h_size_2 = 16
                h_size_4 = 8
                h_size_8 = 4
                h_size_16 = 2

                print("G:", z.get_shape())
                net = linear(z, 512 * h_size_16 * h_size_16, scope='g_fc1' + '_' + self.dataset_name)
                print("G:", net.get_shape())
                net = tf.nn.relu(
                    bn(tf.reshape(net, [self.batch_size, h_size_16, h_size_16, 512]), is_training=is_training,
                       scope='g_bn1')
                )
                print("G:", net.get_shape())
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_8, h_size_8, 256], 5, 5, 2, 2,
                                name='g_dc2' + '_' + self.dataset_name), is_training=is_training, scope='g_bn2')
                )
                print("G:", net.get_shape())
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_4, h_size_4, 128], 5, 5, 2, 2,
                                name='g_dc3' + '_' + self.dataset_name), is_training=is_training, scope='g_bn3')
                )
                print("G:", net.get_shape())
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, h_size_2, h_size_2, 64], 5, 5, 2, 2,
                                name='g_dc4' + '_' + self.dataset_name), is_training=is_training, scope='g_bn4')
                )
                print("G:", net.get_shape())
                out = tf.nn.tanh(
                    deconv2d(net, [self.batch_size, self.output_height, self.output_width, self.c_dim], 5, 5, 2, 2,
                             name='g_dc5' + '_' + self.dataset_name)
                )
                print("G:", out.get_shape())
                print("------------------------")

            return out

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """

        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # labels
        self.y = tf.placeholder(tf.float32, [bs, self.y_dim], name='y')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, self.y, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, self.y, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, self.y, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        self.d_loss_real_vec = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits,
                                                                       labels=tf.ones_like(D_real))
        self.d_loss_fake_vec = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
                                                                       labels=tf.zeros_like(D_fake))

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_optim_init = dp_optimizer.DPGradientDescentGaussianOptimizer(l2_norm_clip=self.l2_norm_clip,
                                                                           noise_multiplier=self.noise_multiplier,
                                                                           num_microbatches=self.batch_size,
                                                                           learning_rate=self.learningRateD)

            global_step = tf.train.get_global_step()

            self.d_optim = d_optim_init.minimize_ours(d_loss_real=self.d_loss_real_vec, d_loss_fake=self.d_loss_fake_vec,
                                                  global_step=global_step, var_list=d_vars)

            self.g_optim = tf.train.AdamOptimizer(self.learningRateG, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)


        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, self.y, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        self.test_labels = self.data_y[0:self.batch_size]

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        epoch = start_epoch
        should_terminate = False
        while (epoch < self.epoch and not should_terminate):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                # for idx in range(0,100):
                batch_images = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_labels = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network

                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                       feed_dict={self.inputs: batch_images, self.y: batch_labels,
                                                                  self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                eps = self.compute_epsilon((epoch * self.num_batches) + idx)

                # For the Moments accountant, we should always have \
                # spent_eps == max_target_eps.
                if (eps > self.epsilon):
                    should_terminate = True
                    print("TERMINATE !! Run out of Privacy Budget.....")
                    epoch = self.epoch
                    break

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                       feed_dict={self.y: batch_labels, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                # print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                #       % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))
                _ = self.sess.run(self.fake_images,
                                  feed_dict={self.z: self.sample_z, self.y: self.test_labels})

                # save training results for every 300 steps
                if np.mod(counter, 10) == 0:
                    print("Iteration : " + str(idx) + " Eps: " + str(eps))
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z, self.y: self.test_labels})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                check_folder(
                                    self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))
            epoch = epoch + 1

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            if (self.dataset_name == 'mnist'):
                self.visualize_results_MNIST(epoch)
            elif (self.dataset_name == 'cifar10'):
                self.visualize_results_CIFAR(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

        # Classification  TO-DO: classifier for cifar100

        def compute_fpr_tpr_roc(Y_test, Y_score):
            n_classes = Y_score.shape[1]
            false_positive_rate = dict()
            true_positive_rate = dict()
            roc_auc = dict()
            for class_cntr in range(n_classes):
                false_positive_rate[class_cntr], true_positive_rate[class_cntr], _ = roc_curve(
                    Y_test[:, class_cntr],
                    Y_score[:, class_cntr])
                roc_auc[class_cntr] = auc(false_positive_rate[class_cntr], true_positive_rate[class_cntr])

            # Compute micro-average ROC curve and ROC area
            false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(Y_test.ravel(),
                                                                                     Y_score.ravel())
            roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

            return false_positive_rate, true_positive_rate, roc_auc

        def classify(X_train, Y_train, X_test, classiferName, random_state_value=0):
            if classiferName == "lr":
                classifier = OneVsRestClassifier(
                    LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                       random_state=random_state_value))
            elif classiferName == "mlp":
                classifier = OneVsRestClassifier(MLPClassifier(random_state=random_state_value, alpha=1))

            elif classiferName == "rf":
                classifier = OneVsRestClassifier(
                    RandomForestClassifier(n_estimators=100, random_state=random_state_value))

            else:
                print("Classifier not in the list!")
                exit()
            Y_score = classifier.fit(X_train, Y_train).predict_proba(X_test)
            return Y_score

        batch_size = int(self.batch_size)

        if (self.dataset_name == "mnist"):

            n_class = np.zeros(10)
            n_class[0] = 5923 - batch_size
            n_class[1] = 6742
            n_class[2] = 5958
            n_class[3] = 6131
            n_class[4] = 5842
            n_class[5] = 5421
            n_class[6] = 5918
            n_class[7] = 6265
            n_class[8] = 5851
            n_class[9] = 5949

            Z_sample = np.random.uniform(-1, 1, size=(batch_size, self.z_dim))
            y = np.zeros(batch_size, dtype=np.int64) + 0
            y_one_hot = np.zeros((batch_size, self.y_dim))
            y_one_hot[np.arange(batch_size), y] = 1
            images = self.sess.run(self.fake_images, feed_dict={self.z: Z_sample, self.y: y_one_hot})

            for classLabel in range(0, 10):
                for _ in range(0, int(n_class[classLabel]), batch_size):
                    Z_sample = np.random.uniform(-1, 1, size=(batch_size, self.z_dim))
                    y = np.zeros(batch_size, dtype=np.int64) + classLabel
                    y_one_hot_init = np.zeros((batch_size, self.y_dim))
                    y_one_hot_init[np.arange(batch_size), y] = 1

                    images = np.append(images, self.sess.run(self.fake_images,
                                                             feed_dict={self.z: Z_sample, self.y: y_one_hot_init}),
                                       axis=0)
                    y_one_hot = np.append(y_one_hot, y_one_hot_init, axis=0)

            X_test, Y_test = loadlocal_mnist(images_path=self.base_dir + '/data/mnist/t10k-images.idx3-ubyte',
                                             labels_path=self.base_dir + '/data/mnist/t10k-labels.idx1-ubyte')

            Y_test = [int(y) for y in Y_test]

            classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            Y_test = label_binarize(Y_test, classes=classes)

        if (self.dataset_name == "cifar10"):
            n_class = np.zeros(10)
            for t in range(1, 10):
                n_class[t] = 1000
            print(len(n_class))
            Z_sample = np.random.uniform(-1, 1, size=(batch_size, self.z_dim))
            y = np.zeros(batch_size, dtype=np.int64) + 0
            y_one_hot = np.zeros((batch_size, self.y_dim))
            y_one_hot[np.arange(batch_size), y] = 1
            images = self.sess.run(self.fake_images, feed_dict={self.z: Z_sample, self.y: y_one_hot})

            for classLabel in range(0, 10):
                for _ in range(0, int(n_class[classLabel]), batch_size):
                    Z_sample = np.random.uniform(-1, 1, size=(batch_size, self.z_dim))
                    y = np.zeros(batch_size, dtype=np.int64) + classLabel
                    y_one_hot_init = np.zeros((batch_size, self.y_dim))
                    y_one_hot_init[np.arange(batch_size), y] = 1

                    images = np.append(images, self.sess.run(self.fake_images,
                                                             feed_dict={self.z: Z_sample, self.y: y_one_hot_init}),
                                       axis=0)
                    y_one_hot = np.append(y_one_hot, y_one_hot_init, axis=0)

            X_test, Y_test = load_cifar10_test()

            # Y_test = [int(y) for y in Y_test]

            classes = range(0, 10)
            Y_test = label_binarize(Y_test, classes=classes)

        print("  Classifying - Logistic Regression...")

        print(np.shape(images))
        print(np.shape(y_one_hot))

        TwoDim_images = images.reshape(np.shape(images)[0], -2)
        print(TwoDim_images.shape)
        X_test = X_test.reshape(np.shape(X_test)[0], -2)
        Y_score = classify(TwoDim_images, y_one_hot, X_test, "lr", random_state_value=30)

        # -------------------------------------------------
        print("  Classifying - Logistic Regression...")
        Y_score = classify(TwoDim_images, y_one_hot, X_test, "lr", random_state_value=30)

        print("  Computing ROC - Logistic Regression ...")
        false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(Y_test, Y_score)

        classification_results_fname = self.base_dir + "CGAN_AuROC.txt"
        classification_results = open(classification_results_fname, "w")

        classification_results.write("\nepsilon : {:.2f}, sigma: {:.2f}, clipping value: {:.2f}".format(
            (self.epsilon),
            round(self.noise_multiplier, 2),
            round(self.l2_norm_clip, 2)))

        classification_results.write("\nAuROC - logistic Regression: " + str(roc_auc["micro"]))
        classification_results.write("\n--------------------------------------------------------------------\n")
        # ------------------------------------------------------------
        print("  Classifying - Random Forest...")
        Y_score = classify(TwoDim_images, y_one_hot, X_test, "rf", random_state_value=30)

        print("  Computing ROC - Random Forest ...")
        false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(Y_test, Y_score)

        #        classification_results_fname = self.base_dir + "CGAN_AuROC.txt"
        #        classification_results = open(classification_results_fname, "w")

        classification_results.write("\nepsilon : {:.2f}, sigma: {:.2f}, clipping value: {:.2f}".format(
            (self.epsilon),
            round(self.noise_multiplier, 2),
            round(self.l2_norm_clip, 2)))

        classification_results.write("\nAuROC - random Forest: " + str(roc_auc["micro"]))
        classification_results.write("\n--------------------------------------------------------------------\n")
        # --------------------------------------------------------------
        # --------------------------------------------------------------
        print("  Classifying - multilayer Perceptron ...")
        Y_score = classify(TwoDim_images, y_one_hot, X_test, "mlp", random_state_value=30)

        print("  Computing ROC - Multilayer Perceptron ...")
        false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(Y_test, Y_score)

        classification_results.write("\nepsilon : {:.2f}, sigma: {:.2f}, clipping value: {:.2f}".format(
            (self.epsilon),
            round(self.noise_multiplier, 2),
            round(self.l2_norm_clip, 2)))

        classification_results.write("\nAuROC - multilayer Perceptron: " + str(roc_auc["micro"]))
        classification_results.write("\n--------------------------------------------------------------------\n")

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def compute_epsilon(self, steps):
        """Computes epsilon value for given hyperparameters."""
        if self.noise_multiplier == 0.0:
            return float('inf')
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        sampling_probability = self.batch_size / 60000
        rdp = compute_rdp(q=sampling_probability,
                          noise_multiplier=self.noise_multiplier,
                          steps=steps,
                          orders=orders)
        # Delta is set to 1e-5 because MNIST has 60000 training points.
        return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

    # CIFAR 10
    def visualize_results_CIFAR(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)  # 64, 100
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))  # 8

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))  # 100, 100

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_matplot_img(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                         self.result_dir + '/' + self.model_dir + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    # MNIST
    def visualize_results_MNIST(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """
        y = np.random.choice(self.y_dim, self.batch_size)
        y_one_hot = np.zeros((self.batch_size, self.y_dim))
        y_one_hot[np.arange(self.batch_size), y] = 1

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

        """ specified condition, random noise """
        n_styles = 10  # must be less than or equal to self.batch_size

        np.random.seed()
        si = np.random.choice(self.batch_size, n_styles)

        for l in range(self.y_dim):
            y = np.zeros(self.batch_size, dtype=np.int64) + l
            y_one_hot = np.zeros((self.batch_size, self.y_dim))
            y_one_hot[np.arange(self.batch_size), y] = 1

            samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})
            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        check_folder(
                            self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_class_%d.png' % l)

            samples = samples[si, :, :, :]

            if l == 0:
                all_samples = samples
            else:
                all_samples = np.concatenate((all_samples, samples), axis=0)

        """ save merged images to check style-consistency """
        canvas = np.zeros_like(all_samples)
        for s in range(n_styles):
            for c in range(self.y_dim):
                canvas[s * self.y_dim + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_images(canvas, [n_styles, self.y_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

