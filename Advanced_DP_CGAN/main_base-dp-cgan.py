import tensorflow as tf
import os
import numpy as np
from gan import Base_DP_CGAN
from gan.utils import show_all_variables

#base_dir = "/content/gdrive/Team Drives/PrivacyGenomics/New-DP-CGAN/"
base_dir = "./"
epoch = 100
epsilons = range(2, 10, 2)
sigmas = np.arange(0.5, 1, 0.1)
clippings = np.arange(0.2, 1, 0.1)

out_dir_init = base_dir +"Base_DP_CGAN_Output/"
if not os.path.exists(out_dir_init):
    os.mkdir(out_dir_init)

for epsilon in epsilons:
    for sigma in sigmas:
        for clipping in clippings:

            print ("\nepsilon : {:d}, sigma: {:.2f}, clipping value: {:.2f}".format(epsilon, round(sigma,2), round(clipping,2)))
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

                out_dir =  out_dir_init +("epsilon_{:d}_sigma_{:.2f}_clip_{:.2f}".format(epsilon, round(sigma,2), round(clipping,2)))

                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)

                cgan = Base_DP_CGAN.Base_DP_CGAN(
                        sess,
                        epoch=epoch,
                        z_dim=100,
                        batch_size=256,
                        sigma=sigma,
                        clipping=clipping,
                        delta=1e-5,
                        epsilon=[epsilon],
                        learning_rate=0.05,
                        dataset_name='mnist',
                        base_dir = base_dir,
                        result_dir=out_dir + "/")

                cgan.build_model()
                print(" [*] Building model finished!")

                cgan.train()
                print(" [*] Training finished!")

                cgan.visualize_results(epoch-1)
                print(" [*] Testing finished!")

            tf.reset_default_graph()
            print("------------------------------------------------------------------------------------------")