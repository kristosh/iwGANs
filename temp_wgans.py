# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
import cv2 
import tensorflow as tf
from keras.callbacks import TensorBoard


def combine_images(generated_images):

    generated_images =  np.transpose(generated_images , (0, 3, 1, 2))

    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((3,(height+3)*shape[0], (width+3)*shape[1]),
                     dtype=generated_images.dtype)

    for index, img in enumerate(generated_images):

        new_shape = (img.shape[0], img.shape[1]+ 4, img.shape[2] + 4)
        img_ = np.zeros(new_shape)
        img_[:, 2:  2+img.shape[1], 2:  2+img.shape[2]] = img

        i = int(index/width)
        j = index % width
        image[:, i*new_shape[1]: (i+1)*new_shape[1], j*new_shape[2]: (j+1)*new_shape[2]] = img_[:, :, :]
    return image


def store_image_maps(images_db, filename):
    image = combine_images(images_db)
    image = image * 127.5 + 127.5
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    cv2.imwrite(filename,image)


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
            
        self.target_mod = "audio"
        self.input_feats = "prime"
        self.learning_param = 0.0001
        self.no_of_trial = "sixth"
        self.no_input_feats = 64

        self.db_path = "../../GANs_models/tmp_dtst/"
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 15
        optimizer = RMSprop(lr=self.learning_param)

        self.obj = data_handle()

        self.batch_size = 32

        self.db_path_rv = "../../GANs_cnn/models/ravdess/ravdess/"
        self.db_path_cr = "../../GANs_cnn/models/"
        
        if self.target_mod == "audio":
            self.img_rows = 28
            self.img_cols = 112
            self.channels = 3
            self.img_shape = (self.img_rows, self.img_cols, self.channels)
            self.latent_dim = 28

            # Build the generator and critic
            self.generator = generator_model(self.latent_dim)
            #self.generator = build_generator_updated(self.latent_dim, self.channels)
            #self.generator = self.build_generator_old()
            self.critic = build_critic(self.img_shape)
        elif self.target_mod == "face":
           
            self.img_rows = 28
            self.img_cols = 28
            self.channels = 3
            self.img_shape = (self.img_rows, self.img_cols, self.channels)
            self.latent_dim = 38 #550

            # Build the generator and critic
            self.generator = build_generator_face(self.latent_dim, self.channels)
            #self.generator = self.build_generator_old()
            self.critic = build_critic(self.img_shape)

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)
        face_img = Input((3, 28, 28)) 
        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator([face_img, z_disc])

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

        self.critic_model = Model(inputs=[real_img, face_img,  z_disc],
                            outputs=[valid, fake, validity_interpolated, aux1])
        
        self.critic_model.compile(loss=[self.wasserstein_loss,
                self.wasserstein_loss,
                partial_gp_loss,
                'categorical_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'],
            loss_weights=[1, 1, 5, 1])
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
        img = self.generator([face_img, z_gen])
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        
        self.generator_model = Model([face_img, z_gen], valid)
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
        
        _dirs_ = os.listdir(self.db_path_cr)
        
        batch_size = self.batch_size
        loss_total = []

        for _dir_ in _dirs_:

            _dct_ = self.obj.get_data_cp('train', self.db_path_cr,  _dir_)

            file_name = self.target_mod
            lbls_train = _dct_["_lbls_trn_"][:,0:6]

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
                +"_" + str(self.latent_dim) \
                +"_" + str(self.learning_param) \
                +"_" + self.no_of_trial \
                +"_gen_noise_feats_generator_model.yaml"
                    
            with open(model_name, "w") as yaml_file:
                yaml_file.write(model_yaml_gn)
            
            with open(model_name, "w") as yaml_file:
                yaml_file.write(model_yaml_cr)

            my_loss = []
            for epoch in range(epochs):
                if epoch == 50000:
                    self.gen_data_iwGANs(_dct_, file_name)

                for _ in range(self.n_critic):
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    # Select a random batch of images
                    idx = np.random.randint(0, _dct_["_trg_trn_"].shape[0], batch_size)
                    imgs = np.transpose(_dct_["_trg_trn_"][idx], (0, 2, 3, 1))
                    batch_lbls = _dct_["_lbls_trn_"][idx]
                    feats = _dct_["_src_trn_"][idx]
                    # Sample generator input
                    
                    noise = np.random.normal(0, 1, (batch_size, 22))
                
                    conditional_vector = np.concatenate([noise, batch_lbls], axis = 1)
                    # Train the critic
                    #pdb.set_trace()
                    d_loss = self.critic_model.train_on_batch([imgs,feats, conditional_vector],[valid, fake, dummy, batch_lbls])
                    my_loss.append(d_loss)
                # ---------------------
                #  Train Generator
                # ---------------------
                g_loss = self.generator_model.train_on_batch([feats,conditional_vector], [valid, batch_lbls])
                
                tensorboard.on_epoch_end(epoch, self.named_logs(self.generator_model, g_loss))           
                print (d_loss)
                #print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss[0]))

                # If at save interval => save generated image samples
                
                if epoch % sample_interval == 0:
                    self.sample_images(epoch, 
                        batch_lbls, 
                        feats, 
                        batch_size, 
                        imgs, 
                        file_name)

                    weight_name = "../../GANs_models/" \
                        +self.input_feats \
                        +"_"+str(self.latent_dim) \
                        +"_"+str(self.learning_param) \
                        +"_"+self.no_of_trial \
                        +"_gen_noise_feats_" \
                        + file_name+"_"

                    self.generator.save_weights(weight_name) 

            _loss_obj_ = "../../GANs_models/" \
                +self.input_feats \
                +"_"+str(self.latent_dim) \
                +"_"+str(self.learning_param) \
                +"_"+self.no_of_trial \
                +"_gen_noise_feats_" \
                + file_name+"_loss.pkl"
            store_obj(_loss_obj_, my_loss)


    # Store generated images.
    def sample_images(self, epoch, batch_lbls, feats, batch_size, imgs, file_name):
        r, c = 5, 5
        #noise = np.random.normal(0, 1, (r * c, self.latent_dim- 6))
        noise = np.random.normal(0, 1, (batch_size, 22))
        conditional_vector = np.concatenate([ noise, batch_lbls], axis = 1)
        #conditional_vector = np.concatenate([noise, batch_lbls], axis = 1)
        gen_imgs = self.generator.predict([feats, conditional_vector])
        # Finally store the images to the local path
        
        store_image_maps(gen_imgs, 
            "../../GANs_assets/generated_imgs/wgans/" \
            +self.input_feats+"_noise_lbls_"  \
            + file_name  \
            +"_new_img_%d.png" % epoch)     
        
        store_image_maps(imgs, 
            "../../GANs_assets/generated_imgs/wgans/" \
            +self.input_feats+"_noise_lbls_"  \
            + file_name  \
            +"_real_img_%d.png" % epoch)  


    # Generate new samples using the trained Generator.
    def gen_data_iwGANs(self, _dct_, file_name): 

        weight_name = "../../GANs_models/" \
            +self.input_feats \
            +"_"+str(self.latent_dim) \
            +"_"+str(self.learning_param) \
            +"_"+self.no_of_trial \
            +"_gen_noise_feats_" \
            + file_name+"_"

        self.generator.load_weights(weight_name)        
        
        noise = np.random.normal(0, 1, (_dct_["test_feats"].shape[0], 32))
        
        noise = np.concatenate([_dct_["test_feats"], 
            noise, 
            _dct_["lbls_test"][:,0:6]], 
            axis = 1)
        
        gen_train = self.generator.predict([noise])
        pdb.set_trace()
        for index in range(0, 2):
            
            gen_data = {"gen_train": gen_train[75000*index:(75000+ 75000*index)], 
                "lbls_train": _dct_["lbls_test"][75000*index:(75000+ 75000*index)]}

            stored_name = "../../GANs_models/" \
                +self.input_feats \
                +"_gen_audio_face_feat_" \
                +str(index) \
                +".pkl"

            self.obj.store_obj(stored_name, gen_data)
            gen_data = None  


if __name__ == '__main__':

    wgan = WGANGP()
    wgan.train(epochs=50000, 
            batch_size=32, 
            sample_interval=100)