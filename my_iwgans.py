# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial
from sklearn.utils import shuffle
from keras.models import model_from_yaml

import keras.backend as K
import tensorflow.keras as keras
import matplotlib.pyplot as plt

import sys
import math
import numpy as np

from data_handling import *
from keras_models import *
from _mtds_fWGANs_ import *

import cv2 
import tensorflow as tf
from keras.callbacks import TensorBoard

# batch size for the whole network.
batch_size = 64

# it is used for the implemenaiton of gradient penalty for the Wasserstein loss.
class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        
        self.mtds = MTDS()

        self.target_mod = "face"
        self.input_feats = "3dCNN"
        self.db = "creamad"
        self.learning_param = 0.0001
        self.input_type = "without_source"
        self._featsD = 0
        self._noizeD = 32

        self.face_sequence = False
        self.comment = ""
        self.db_path = "../../GANs_models/tmp_dtst/dataAugm/"
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 15
        optimizer = RMSprop(lr=self.learning_param)

        if self.input_type == "with_source":
            self.input_to_G = True
        else:
            self.input_to_G = False

        self.obj = data_handle()
        
        if self.target_mod == "audio":
            self.img_rows = 28
            self.img_cols = 112
            self.channels = 3
            self.img_shape = (self.img_rows, self.img_cols, self.channels)
            self.latent_dim = 296

            # Build the generator and critic
            self.generator = build_generator(self.latent_dim, self.channels)
            #self.generator = self.build_generator_old()
            self.critic = build_critic(self.img_shape)

        elif self.target_mod == "face":
           
            if self.face_sequence == False:
                self.img_rows = 28
                self.img_cols = 28
                self.channels = 3
            else:
                self.img_rows = 28
                self.img_cols = 280
                self.channels = 3

            self.img_shape = (self.img_rows, self.img_cols, self.channels)
            self.latent_dim = self._featsD + self._noizeD + 6

            # Build the generator and critic
            self.generator = build_generator_face_v(self.latent_dim, self.channels, self.face_sequence)
            #self.generator = self.build_generator_old()
            self.critic = build_critic_v(self.img_shape, self.face_sequence)

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake, aux1 = self.critic(fake_img)
        valid, aux2 = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated, aux3 = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated, aux1])
        
        self.critic_model.compile(loss=[self.wasserstein_loss,
                self.wasserstein_loss,
                partial_gp_loss,
                'categorical_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'],
            loss_weights=[1, 1, 3, 5])

        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    
    def named_logs(self, model, logs):
        
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result
    
    def load_model_from_yaml(self, file_name):

        yaml_model = open(file_name, 'r')
        loaded_model_yaml = yaml_model.read()
        yaml_model.close()
        loaded_model = model_from_yaml(loaded_model_yaml)

        return loaded_model


    def train(self, epochs, batch_size, sample_interval=50):
        
        # my_model = self.load_model_from_yaml(
        #     "../../GANs_models/"
        #     +self.input_feats
        #     +"_gen_noise_feats_generator_model.yaml")

        if self.input_feats == "3dCNN":

            if self.db == "ravdess":
                _dct_ = self.obj.load_3d_dataset_rav(self.target_mod) #ravdess
            else:
                if self.input_to_G == True:
                    _dct_ = self.obj.load_3d_dataset_v2(self.target_mod, self.img_rows) #crema
                else:
                    _dct_ = self.obj.load_3d_dataset(self.target_mod, self.img_rows)
        else:
            _dct_ = self.obj.temporal_feats(self.target_mod, 1, 
                self._featsD, 
                self.input_feats, 
                self.db_path)

        file_name = self.target_mod
        lbls_train = _dct_["trn_lbls"] 

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        
        try:
            del valid_target
            del test_target
        except Exception as e:
            print("Temporal version")

        # Create the TensorBoard callback,
        # which we will drive manually
        tensorboard = keras.callbacks.TensorBoard(
            log_dir='my_tf_logs',
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=True
        )

        tensorboard.set_model(self.generator_model)

        model_yaml_gn = self.generator_model.to_yaml()
        model_yaml_cr = self.critic_model.to_yaml()

        model_name = "../../GANs_models/" \
            +self.input_feats \
            +"_" + self.db \
            +"_" + str(self.latent_dim) \
            +"_" + str(self.learning_param) \
            +"_"+str(self.input_feats) \
            +"_" + self.input_type \
            +"_" + self.comment \
            +"_"+str(self.img_rows) \
            +"_gen_noise_feats_generator_model.yaml"
                  
        with open(model_name, "w") as yaml_file:
            yaml_file.write(model_yaml_gn)
        
        with open(model_name, "w") as yaml_file:
            yaml_file.write(model_yaml_cr)

        my_loss = []
        self.mtds._gen_datwGANs(_dct_, file_name, self)
        
        for epoch in range(epochs):
            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Select a random batch of images
                idx = np.random.randint(0, _dct_["trn_trg"].shape[0], batch_size)
                imgs = self.mtds._cmb_ten_fr_(_dct_["trn_trg"][idx])
                batch_lbls = _dct_["trn_lbls"][idx]
                feats = _dct_["trn_fts"][idx]
                
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self._noizeD))

                if self.input_to_G ==True:
                    conditional_vector = np.concatenate([feats, noise, batch_lbls], axis = 1)
                else:
                    conditional_vector = np.concatenate([noise, batch_lbls], axis = 1)
                    #conditional_vector = np.concatenate([noise], axis = 1)
                
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, conditional_vector],[valid, fake, dummy, batch_lbls])
                my_loss.append(d_loss)
            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.generator_model.train_on_batch(conditional_vector, [valid, batch_lbls])
            
            tensorboard.on_epoch_end(epoch, self.named_logs(self.generator_model, g_loss))           
            print (d_loss)
            #print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss[0]))

            # If at save interval => save generated image samples
            
            if epoch % 500 == 0:
                self.sample_images(epoch, 
                    batch_lbls, 
                    feats, 
                    batch_size, 
                    imgs, 
                    file_name)

                weight_name = "../../GANs_models/" \
                    +self.input_feats \
                    +"_" + self.db \
                    +"_"+str(self.latent_dim) \
                    +"_"+str(self.learning_param) \
                    +"_"+str(self.input_feats) \
                    +"_"+self.input_type \
                    +"_" + self.comment \
                    +"_"+str(self.img_rows) \
                    +"_gen_noise_feats_" \
                    + file_name+"_"

                self.generator.save_weights(weight_name) 

        _loss_obj_ = "../../GANs_models/loss_" \
            +self.input_feats \
            +"_" + self.db \
            +"_"+str(self.latent_dim) \
            +"_"+str(self.learning_param) \
            +"_"+str(self.input_feats) \
            +"_"+self.input_type \
            +"_" + self.comment \
            +"_"+str(self.img_rows) \
            +"_gen_noise_feats_" \
            + file_name+"_loss.pkl"

        self.obj.store_obj(_loss_obj_, my_loss)


    # Store generated images.
    def sample_images(self, epoch, batch_lbls, feats, batch_size, imgs, file_name):
        r, c = 5, 5
        #noise = np.random.normal(0, 1, (r * c, self.latent_dim- 6))
        noise = np.random.normal(0, 1, (batch_size, self._noizeD))
        
        if self.input_to_G == True:
            conditional_vector = np.concatenate([feats, noise, batch_lbls], axis = 1)
        else:
            conditional_vector = np.concatenate([noise, batch_lbls], axis = 1)
            #conditional_vector = np.concatenate([noise], axis = 1)

        #conditional_vector = np.concatenate([noise, batch_lbls], axis = 1)
        gen_imgs = self.generator.predict(conditional_vector)
        # Finally store the images to the local path
        
        self.mtds._str_imgs_(gen_imgs, 
            "../../GANs_assets/generated_imgs/wgans/" \
            +self.input_feats+"_noise_lbls_"  \
            +"_" + self.db \
            +"_"+str(self.input_feats) +"_"\
            + file_name \
            +"_"+self.input_type \
            +"_" + self.comment \
            +"_"+str(self.img_rows) \
            +"_new_img_%d.png" % epoch)     
        
        self.mtds._str_imgs_(imgs, 
            "../../GANs_assets/generated_imgs/wgans/" \
            +self.input_feats+"_noise_lbls_"  \
            +self.db \
            +"_"+str(self.input_feats) \
            + file_name  \
            +"_"+self.input_type \
            +"_" + self.comment \
            +"_"+str(self.img_rows) \
            +"_real_img_%d.png" % epoch)  
        
        fl = "../../GANs_assets/metrics/wgans/3dCNN/" \
            +self.input_feats+"_noise_lbls_"  \
            + str(self.input_feats) \
            +"_" + self.db + "_" \
            + file_name  \
            +"_"+self.input_type \
            +"_" + self.comment \
            +"_"+str(self.img_rows) \
            +"_quality_metrics_data_%d.pkl" % epoch

        my_gen_dict = {"real": imgs, "generated": gen_imgs, "labels": batch_lbls}
        self.obj.store_obj(fl, my_gen_dict)


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=50000, 
        batch_size=batch_size, 
        sample_interval=100)