"""
Reference for Vanilla GAN code:
https://github.com/wiseodd/generative-models/blob/master/GAN/

Reference for Moment accountant code:
https://github.com/tensorflow/models/tree/master/research/differential_privacy
"""

# base directory for the project
baseDir = "../"

#Import the requiered python packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

#Import required Differential Privacy packages
sys.path.append(baseDir);

from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from differential_privacy.dp_sgd.dp_optimizer import sanitizer
from differential_privacy.dp_sgd.dp_optimizer import utils
from differential_privacy.privacy_accountant.tf import accountant

def xavier_init(size):
    """ Xavier Function to keep the scale of the gradients roughly the same
        in all the layers. 
    """
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
  

#Initializations for a two-layer discriminator network
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

#Initializations for a two-layer genrator network
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
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        #plt.show()
    return fig

def del_all_flags(FLAGS):
    """ Function to delete all flags before declare
    """
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

#Delete all Flags
del_all_flags(tf.flags.FLAGS)

# Set training parameters
tf.flags.DEFINE_string('f', '', 'kernel')
tf.flags.DEFINE_float("lr",0.05, "start learning rate")
tf.flags.DEFINE_float("end_lr", 0.05, "end learning rate")
tf.flags.DEFINE_float("lr_saturate_epochs", 0,
                      "learning rate saturate epochs; set to 0 for a constant"
                      "learning rate of --lr.")
tf.flags.DEFINE_integer("batch_size", 600, "The training batch size.")
tf.flags.DEFINE_integer("batches_per_lot", 1, "Number of batches per lot.")
tf.flags.DEFINE_integer("num_training_steps", 100000, "The number of training"
                        "steps. This counts number of lots.")

#Flags that control privacy spending during training
tf.flags.DEFINE_float("target_delta",1e-5,"Maximum delta for"
                      "--terminate_based_on_privacy.")
tf.flags.DEFINE_float("sigma", 2 , "Noise sigma, used only if accountant_type"
                      "is Moments")
tf.flags.DEFINE_string("target_eps", "9.6",
                       "Log the privacy loss for the target epsilon's. Only"
                       "used when accountant_type is Moments.")
tf.flags.DEFINE_float("default_gradient_l2norm_bound", 4, "norm clipping")


FLAGS = tf.flags.FLAGS

# Set accountant type to GaussianMomentsAccountant
NUM_TRAINING_IMAGES = 60000
priv_accountant = accountant.GaussianMomentsAccountant(NUM_TRAINING_IMAGES)

#Sanitizer
batch_size = FLAGS.batch_size
clipping_value = FLAGS.default_gradient_l2norm_bound 
gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(priv_accountant,
                    [clipping_value / batch_size, True])

#Instantiate the Generator Network
G_sample = generator(Z, y)

#Instantiate the Discriminator Network
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

# Discriminator loss for real data
D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(\
                                            logits=D_logit_real,\
                                            labels=tf.ones_like(D_logit_real)),\
                                             [0])
# Discriminator loss for fake data
D_loss_fake = tf.reduce_mean(\
                             tf.nn.sigmoid_cross_entropy_with_logits(\
                             logits=D_logit_fake,\
                             labels=tf.zeros_like(D_logit_fake)),[0])

# Generator loss 
G_loss = tf.reduce_mean(\
                        tf.nn.sigmoid_cross_entropy_with_logits(\
                        logits=D_logit_fake, labels=tf.ones_like(D_logit_fake))\
                        ,[0])
                                                                                                
# Generator optimizer
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

# Discriminator Optimizer
#------------------------------------------------------------------------------
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
D_solver = dp_optimizer.DPGradientDescentOptimizer(\
                                                   lr,[None, None],\
                                                   gaussian_sanitizer,\
                                                   sigma=sigma,\
                                                   batches_per_lot=\
                                                   FLAGS.batches_per_lot).\
                                                   minimize_ours(\
                                                            D_loss_real,\
                                                            D_loss_fake ,\
                                                            var_list=theta_D)
#------------------------------------------------------------------------------

# Set output directory
resultDir = baseDir + "/conditional-dpgan-mnist/results"
if not os.path.exists(resultDir):
   os.makedirs(resultDir)
    
resultPath = resultDir + "/bs_{}_s_{}_c_{}_d_{}_e_{}".format(\
                              batch_size,\
                              sigma,\
                              clipping_value,\
                              FLAGS.target_delta, FLAGS.target_eps)

if not os.path.exists(resultPath):
   os.makedirs(resultPath)


#Delete all Flags
del_all_flags(tf.flags.FLAGS)

# Set training parameters
tf.flags.DEFINE_string('f', '', 'kernel')
tf.flags.DEFINE_float("lr",0.05, "start learning rate")
tf.flags.DEFINE_float("end_lr", 0.05, "end learning rate")
tf.flags.DEFINE_float("lr_saturate_epochs", 0,
                      "learning rate saturate epochs; set to 0 for a constant"
                      "learning rate of --lr.")
tf.flags.DEFINE_integer("batch_size", 600, "The training batch size.")
tf.flags.DEFINE_integer("batches_per_lot", 1, "Number of batches per lot.")
tf.flags.DEFINE_integer("num_training_steps", 100000, "The number of training"
                        "steps. This counts number of lots.")

#Flags that control privacy spending during training
tf.flags.DEFINE_float("target_delta",1e-5,"Maximum delta for"
                      "--terminate_based_on_privacy.")
tf.flags.DEFINE_float("sigma", 2 , "Noise sigma, used only if accountant_type"
                      "is Moments")
tf.flags.DEFINE_string("target_eps", "9.6",
                       "Log the privacy loss for the target epsilon's. Only"
                       "used when accountant_type is Moments.")
tf.flags.DEFINE_float("default_gradient_l2norm_bound", 4, "norm clipping")


FLAGS = tf.flags.FLAGS

# Set accountant type to GaussianMomentsAccountant
NUM_TRAINING_IMAGES = 60000
priv_accountant = accountant.GaussianMomentsAccountant(NUM_TRAINING_IMAGES)

#Sanitizer
batch_size = FLAGS.batch_size
clipping_value = FLAGS.default_gradient_l2norm_bound 
gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(priv_accountant,
                    [clipping_value / batch_size, True])

#Instantiate the Generator Network
G_sample = generator(Z, y)

#Instantiate the Discriminator Network
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

# Discriminator loss for real data
D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(\
                                            logits=D_logit_real,\
                                            labels=tf.ones_like(D_logit_real)),\
                                             [0])
# Discriminator loss for fake data
D_loss_fake = tf.reduce_mean(\
                             tf.nn.sigmoid_cross_entropy_with_logits(\
                             logits=D_logit_fake,\
                             labels=tf.zeros_like(D_logit_fake)),[0])

# Generator loss 
G_loss = tf.reduce_mean(\
                        tf.nn.sigmoid_cross_entropy_with_logits(\
                        logits=D_logit_fake, labels=tf.ones_like(D_logit_fake))\
                        ,[0])
                                                                                                
# Generator optimizer
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

# Discriminator Optimizer
#------------------------------------------------------------------------------
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
D_solver = dp_optimizer.DPGradientDescentOptimizer(\
                                                   lr,[None, None],\
                                                   gaussian_sanitizer,\
                                                   sigma=sigma,\
                                                   batches_per_lot=\
                                                   FLAGS.batches_per_lot).\
                                                   minimize_ours(\
                                                            D_loss_real,\
                                                            D_loss_fake ,\
                                                            var_list=theta_D)
#------------------------------------------------------------------------------

# Set output directory
resultDir = baseDir + "/conditional-dpgan-mnist/results"
if not os.path.exists(resultDir):
   os.makedirs(resultDir)
    
resultPath = resultDir + "/bs_{}_s_{}_c_{}_d_{}_e_{}".format(\
                              batch_size,\
                              sigma,\
                              clipping_value,\
                              FLAGS.target_delta, FLAGS.target_eps)

if not os.path.exists(resultPath):
   os.makedirs(resultPath)



target_eps = [float(s) for s in FLAGS.target_eps.split(",")]
max_target_eps = max(target_eps)
#mnist = input_data.read_data_sets(baseDir + "/conditional-dpgan-mnist/mnist_dataset", one_hot=True)

#Main Session
with tf.train.SingularMonitoredSession() as sess:
    lot_size = FLAGS.batches_per_lot * batch_size
    lots_per_epoch = NUM_TRAINING_IMAGES / lot_size
    step=0
    
    # Is true when the spent privacy budget exceeds the target budget
    should_terminate = False
    
    #Main loop 
    while (step < FLAGS.num_training_steps and should_terminate==False ):
      
        epoch = step / lots_per_epoch
        curr_lr = utils.VaryRate(FLAGS.lr, FLAGS.end_lr,\
                                 FLAGS.lr_saturate_epochs, epoch)

        for _ in range(FLAGS.batches_per_lot):


            # Save the generated images every 100 steps 
            if step % 100 == 0:

                n_sample = 16
                Z_sample = sample_Z(n_sample, Z_dim)
                y_sample = np.zeros(shape=[n_sample, y_dim])
                y_sample[:, 7] = 1
                samples = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_sample})
                print(samples.shape)
                fig = plot(samples)
                plt.savefig(
                    (resultPath + "/step_{}.png").format(str(step).zfill(3)),
                    bbox_inches='tight')
                plt.close(fig)
                print('Step: {}'.format(step))

            X_mb, y_mb = mnist.train.next_batch(batch_size, shuffle=True)
            Z_sample = sample_Z(batch_size, Z_dim)

            #Update the discriminator network
            _,D_loss_curr,_ = sess.run([D_solver,D_loss_real, D_loss_fake], \
                                      feed_dict={X: X_mb,\
                                                 Z: Z_sample,\
                                                 y: y_mb,\
                                                 lr: curr_lr})
            #Update the generator network
            _, G_loss_curr = sess.run([G_solver, G_loss],
                                      feed_dict={Z: Z_sample, y:y_mb})
            

        # Flag to terminate based on target privacy budget
        terminate_spent_eps_delta = priv_accountant.get_privacy_spent(sess,\
                                    target_eps=[max_target_eps])[0]

        # For the Moments accountant, we should always have \
        # spent_eps == max_target_eps.
        if (terminate_spent_eps_delta.spent_delta > FLAGS.target_delta or \
            terminate_spent_eps_delta.spent_eps > max_target_eps):  
            spent_eps_deltas = priv_accountant.get_privacy_spent(\
                               sess, target_eps=target_eps)
            print("TERMINATE!!!!")
            print("Termination Step : " + str(step))
            should_terminate = True
        step=step+1


