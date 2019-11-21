# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
        self.input_feats = "3dCNN"

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.0001)
        
        if self.target_mod == "audio":
            self.img_rows = 28
            self.img_cols = 112
            self.channels = 3
            self.img_shape = (self.img_rows, self.img_cols, self.channels)
            self.latent_dim = 102

            # Build the generator and critic
            self.generator = self.build_generator()
            #self.generator = self.build_generator_old()
            self.critic = self.build_critic()
        elif self.target_mod == "face":
           
            self.img_rows = 28
            self.img_cols = 28
            self.channels = 3
            self.img_shape = (self.img_rows, self.img_cols, self.channels)
            self.latent_dim = 38 #550

            # Build the generator and critic
            self.generator = self.build_generator_face()
            #self.generator = self.build_generator_old()
            self.critic = self.build_critic()

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
    
    def build_generator_old(self):

        inputs = Input((self.latent_dim,))
        e0 = Dense(128 * 7 * 7, activation="relu")(inputs)
        e1 = Reshape((7, 7, 128))(e0)
        e2 = UpSampling2D()(e1)
        
        e3 = Convolution2D(128, 4, 4, activation='linear',init='uniform', border_mode='same')(e2)
        e3 = BatchNormalization(momentum = 0.8)(e3)
        e3 =  Activation('relu')(e3)
        e3 = UpSampling2D()(e3)
        
        e4 = Convolution2D(64, 4, 4, activation='linear',init='uniform', border_mode='same')(e3)
        e4 = BatchNormalization(momentum = 0.8)(e4)
        e4 =  Activation('relu')(e4)
        e4 = UpSampling2D(size=(1, 2))(e4)

        e5 = Convolution2D(64, 4, 4, activation='linear',init='uniform', border_mode='same')(e4)
        e5 = BatchNormalization(momentum = 0.8)(e5)
        e5 =  Activation('relu')(e5)
        e5 = UpSampling2D(size=(1, 2))(e5)

        e6 = Convolution2D(3, 4, 4, activation='linear',init='uniform', border_mode='same')(e5)
        e6 = Activation("tanh")(e6)

        model = Model(input=inputs, output=e6)
        return model


    def build_generator_face(self):

        model = Sequential()
        #pdb.set_trace()
        model.add(Dense(128 * 7 * 7, activation="relu", input_shape=(None, self.latent_dim))) #input_dim
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        #model.add(UpSampling2D(size=(1, 2)))
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        #model.add(UpSampling2D(size=(1, 2)))
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        mdl = Model(noise, output = img)

        return mdl

    def build_generator(self):

        model = Sequential()
        #pdb.set_trace()
        model.add(Dense(128 * 7 * 7, activation="relu", input_shape=(None, self.latent_dim))) #input_dim
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=(1, 2)))
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=(1, 2)))
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu")) 
        #

        # model.add(Conv2D(64, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))

        # model.add(Conv2D(64, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))

        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        mdl = Model(noise, output = img)

        return mdl

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Flatten())

        image = Input(shape=self.img_shape)
        features = model(image)
        
        fake = Dense(1, activation='linear', name='generation')(features)
        aux = Dense(6, activation='softmax', name='auxiliary')(features)

        return Model(input = [image], output=[fake, aux])

    
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
        
        my_model = self.load_model_from_yaml("../../GANs_models/"+self.input_feats+"_gen_noise_feats_generator_model.yaml")
        print(my_model.summary())

        if self.input_feats == "3dCNN":
            (train_feats, train_target, lbls_train, valid_feats, valid_target, lbls_valid, test_feats, test_target, lbls_test) \
            = load_3d_dataset(self.target_mod)
        else:
            train_feats, valid_feats, test_feats, train_target, valid_target, test_target, lbls_train, lbls_valid, lbls_test = lstm_data(self.target_mod, 1)

        file_name = self.target_mod
        lbls_train = lbls_train[:,0:6]
      
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty


        self.gen_data_iwGANs(train_feats, valid_feats, test_feats, lbls_train, lbls_valid, lbls_test) 
        pdb.set_trace()

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

        with open("../../GANs_models/"+self.input_feats+"_gen_noise_feats_generator_model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml_gn)
        
        with open("../../GANs_models/"+self.input_feats+"_gen_noise_feats_critic_model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml_cr)

        for epoch in range(epochs):

            if epoch == 55000:
                self.gen_data_iwGANs(train_feats, valid_feats, test_feats, lbls_train, lbls_valid, lbls_test)

            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Select a random batch of images
                idx = np.random.randint(0, train_target.shape[0], batch_size)
                imgs = train_target[idx]
                batch_lbls = lbls_train[idx]
                feats = train_feats[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, 32))
            
                conditional_vector = np.concatenate([feats, noise, batch_lbls], axis = 1)
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, conditional_vector],[valid, fake, dummy, batch_lbls])
            

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(conditional_vector, [valid, batch_lbls])
            
            tensorboard.on_epoch_end(epoch, self.named_logs(self.generator_model, g_loss))           
            print (d_loss)
            #print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss[0]))

            # If at save interval => save generated image samples
            
            if epoch % sample_interval == 0:
                self.sample_images(epoch, batch_lbls, feats, batch_size, file_name)
                self.generator.save_weights("../../GANs_models/"+self.input_feats+"_gen_noise_feats_"+ file_name+'_') 


    def sample_images(self, epoch, batch_lbls, feats, batch_size, file_name):
        r, c = 5, 5
        #noise = np.random.normal(0, 1, (r * c, self.latent_dim- 6))
        #pdb.set_trace()
        noise = np.random.normal(0, 1, (batch_size, 32))
        conditional_vector = np.concatenate([feats, noise, batch_lbls], axis = 1)
        #conditional_vector = np.concatenate([noise, batch_lbls], axis = 1)
        gen_imgs = self.generator.predict(conditional_vector)
        store_image_maps(gen_imgs, "../../GANs_assets/generated_imgs/wgans/"+self.input_feats+"_noise_lbls_" + file_name +"_new_img_%d.png" % epoch)     


    def gen_data_iwGANs(self, train_feats, valid_feats, test_feats, lbls_train, lbls_valid, lbls_test): 
        
        self.generator.load_weights("../../GANs_models/"+self.input_feats+"_gen_noise_feats_"+self.target_mod+"_")        
        
        noise = np.random.normal(0, 1, (train_feats.shape[0], 32))
        noise = np.concatenate([train_feats, noise, lbls_train[:,0:6]], axis = 1) #noise = face_imgs
        #noise = lbls_train[:,0:6]
        gen_train = self.generator.predict([noise])
   
        gen_data1 = {"gen_train": gen_train[:75000], "lbls_train": lbls_train[:75000]}
        gen_data2 = {"gen_train": gen_train[75001:150000], "lbls_train": lbls_train[75001:150000]}
        gen_data3 = {"gen_train": gen_train[150001:225000], "lbls_train": lbls_train[150001:225000]}
        gen_data4 = {"gen_train": gen_train[225000:], "lbls_train": lbls_train[225000:]}

        store_obj("../../GANs_models/"+self.input_feats+"_gen_audio_face_feat_1.pkl", gen_data1)
        store_obj("../../GANs_models/"+self.input_feats+"_gen_audio_face_feat_2.pkl", gen_data2)
        store_obj("../../GANs_models/"+self.input_feats+"_gen_audio_face_feat_3.pkl", gen_data3)
        store_obj("../../GANs_models/"+self.input_feats+"_gen_audio_face_feat_4.pkl", gen_data4)
        pdb.set_trace()

if __name__ == '__main__':

    wgan = WGANGP()
    wgan.train(epochs=50000, batch_size=32, sample_interval=100)