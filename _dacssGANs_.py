import numpy as np
import glob, pickle
import os, sys

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import cv2
from keras.optimizers import SGD, Adagrad
from PIL import Image
from sklearn.utils import shuffle
from keras import backend as K
import math
import pylab as plt

K.set_image_dim_ordering('th') 
import pdb
def cls(): os.system('clear')

import matplotlib.image
from skimage.transform import resize
from normalization import BatchNormGAN
from data_handling import data_handle
from keras_models import *
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config= config)


class dacssGANs():

    def __init__(self):
        
        self.img_rows = 28
        self.img_cols = 28
        self.in_ch = 3
        self.batch_size = 32
        self.source_modality = "face"
        self.mod = "audio"

        self.db = "RAVDESS"
        self.db_path = "../../GANs_cnn/models/ravdess/trnNseq/"

        self.exp_tp = "with_lbls"
        self.temporal = False

        self.images = 1
        self.d_optim = Adagrad(lr=0.00001)
        self.g_optim = Adagrad(lr=0.00001)
        self.c_optim = Adagrad(lr=0.00001)

        self.mean = 0
        self.var = 0.1
        self.sigma = self.var**0.5

        self.models_path = "../../GANs_models/old_models/"
        self.loss_path =  "../../GANs_assets/training_GANs/"

        self.hndl_obj = data_handle()
        self.input_dim = 108


    def _G_with_D_and_Q_(self, 
            generator, 
            discriminator, 
            classifier, 
            temporal):
        
        if temporal == False:
            inputs = Input((self.in_ch, self.img_cols, self.img_rows))
            input_conditional = Input(shape=(self.input_dim,))
            x_generator = generator([inputs, input_conditional])
        else: 
            inputs = Input(shape=(170,))
            x_generator = generator(inputs)

        #merged = merge([inputs, x_generator], mode='concat', concat_axis=1)
        discriminator.trainable = False

        x_discriminator = discriminator(x_generator)

        classifier.trainable = False
        x_classifier = classifier(x_generator)

        if temporal == False:
            model = Model(input=[inputs, input_conditional], 
                output=[x_generator, x_discriminator, x_classifier])
        else:
            model = Model(input=[inputs], output=[x_generator, x_discriminator, x_classifier])

        return model
    

    def augmented_noise(self, image_batch):

        num, row, col, ch= image_batch.shape

        gauss1 = np.random.normal(self.mean, self.sigma,(num, row, col, ch))
        gauss1 = gauss1.reshape(num, row,col,ch)
        gauss1 = 0.01*gauss1
        gauss1 = gauss1 +np.amax(gauss1)
        

        gauss2 = np.random.normal(self.mean, self.sigma,(num, row, col, ch))
        gauss2 = gauss2.reshape(num, row, col, ch)
        gauss2 = 0.01*gauss2
        gauss2 = gauss2 +np.amax(gauss2)

        return gauss1, gauss2


    def combine_images(self, generated_images):

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


    def store_image_maps(self, images_db, filename):
        image = self.combine_images(images_db)
        image = image * 127.5 + 127.5
        image = np.swapaxes(image, 0, 1)
        image = np.swapaxes(image, 1, 2)
        cv2.imwrite(filename,image)


    def discriminator_loss(self, y_true,y_pred):
        return K.mean(K.binary_crossentropy(K.flatten(y_pred), 
                K.concatenate([K.zeros_like(K.flatten(y_pred[:self.batch_size,:,:,:])),
                K.ones_like(K.flatten(y_pred[:self.batch_size,:,:,:])) ]) ), 
                axis=-1)


    def discriminator_on_generator_loss(self, y_true,y_pred):
        
        return K.mean(K.binary_crossentropy(K.flatten(y_pred), 
                K.ones_like(K.flatten(y_pred))), 
                axis=-1)


    def generator_l1_loss(self, y_true,y_pred):
        return K.mean(K.abs(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)
    

    def _fl_in_dir_(self, _dir_):

        return os.listdir(_dir_)


    def train(self): 
        
        _dirs_ = self._fl_in_dir_(self.db_path)
        batch_size = self.batch_size
        loss_total = []

        for _dir_ in _dirs_:

            if self.temporal == True:
                _dct_ = self.hndl_obj.load_3d_dataset(self.mod, self.db)
            else:
                _dct_ = self.hndl_obj.get_data('train', self.db_path,  _dir_)

            discriminator = discriminator_model()
            
            if self.temporal == False:
                generator = generator_model(self.input_dim)
            else:
                generator = generator_model_temporal()

            classifier = lenet_classifier_model(_dct_["_lbls_trn_"].shape[1])

            discriminator_and_classifier_on_generator = self._G_with_D_and_Q_(
                        generator, 
                        discriminator, 
                        classifier, 
                        self.temporal)
            
            generator.compile(loss=self.generator_l1_loss, optimizer=self.g_optim)

            discriminator_and_classifier_on_generator.compile(
                loss=[self.generator_l1_loss, 
                      self.discriminator_on_generator_loss, 
                      "categorical_crossentropy"],
                optimizer="rmsprop")

            discriminator.trainable = True
            discriminator.compile(loss=self.discriminator_loss, optimizer=self.d_optim) # rmsprop
            classifier.trainable = True

            classifier.compile(loss="categorical_crossentropy", optimizer=self.c_optim, metrics=['accuracy'])
            # classifier.fit(_dct_["_trg_trn_"], _dct_["_lbls_trn_"], epochs = 50, verbose=1)

            classifier.load_weights(self.models_path+'classifier_56x256_'+self.mod)
            # classifier.save_weights(self.models_path+'classifier_56x256_'+self.mod, True)
            # c_loss = classifier.evaluate(_dct_["_trg_trn_"], _dct_["_lbls_trn_"]) 

            # generator.save_weights(self.models_path+
            #     'gen_tmp_dacssGANs_'
            #     +self.mod
            #     +"_"
            #     +self.exp_tp)

            # discriminator.save_weights(self.models_path
            #     +'discr_tmp_dacssGANs_'
            #     +self.mod 
            #     +"_"
            #     +self.exp_tp)

            self.input_dim = 100 + _dct_["_lbls_trn_"].shape[1]

            for epoch in range(0, 500):
                
                if epoch == 300:
                    loss_name = self.loss_path\
                        +"training_loss_for_GANS_"\
                        +self.mod\
                        + "_" + self.db\
                        +"_"+self.exp_tp+"_"\
                        + ".pkl"     
                    self.hndl_obj.store_obj(loss_name, loss_total)

                print("Epoch is", epoch)
                print("Number of batches", int(_dct_["_src_trn_"].shape[0] / batch_size))
                for index in range(int(_dct_["_src_trn_"].shape[0] / batch_size)):
                    
                    source_image_batch = _dct_["_src_trn_"][index * batch_size:(index + 1) * batch_size]
                    image_batch = _dct_["_trg_trn_"][index * batch_size:(index + 1) * batch_size]
                    label_batch = _dct_["_lbls_trn_"][index * batch_size:(index + 1) * batch_size]  # replace with your data here

                    # self.store_image_maps(source_image_batch, "tempFaces.jpg") 
                    # self.store_image_maps(image_batch, "tempAudio.jpg") 
                    if self.temporal == False:
                        noise = np.random.normal(0, 1, (batch_size, 100))
                        noise = np.concatenate([noise, label_batch], axis = 1)
                        generated_images = generator.predict([source_image_batch, noise])
                    else:
                        noise = np.random.normal(0, 1, (batch_size, 100))
                        noise = np.concatenate([noise, label_batch, source_image_batch], axis = 1)
                        generated_images = generator.predict(noise)

                    # Create a function for it.
                    #image_batch = np.transpose(image_batch, (0, 2, 3, 1))
                    gauss1, gauss2 = self.augmented_noise(image_batch)
                    image_batch = image_batch + gauss1
                    generated_images = generated_images +gauss2

                    if index % 2000 == 0:       
                        file_name = "../../GANs_assets/generated_imgs/"\
                            +self.mod\
                            +"/generated"\
                            + "_" + self.db\
                            +str(epoch)+"_"\
                            +"_"+self.exp_tp+"_"\
                            +str(index)+".png"

                        self.store_image_maps(generated_images, file_name)               
                        file_name = "../../GANs_assets/generated_imgs/"\
                            +self.mod+"/target_"\
                            + "_" + self.db\
                            +str(epoch)\
                            +"_"+str(index)\
                            +"_"+self.exp_tp+"_"\
                            +".png"

                        self.store_image_maps(image_batch, file_name)

                    # Training D:
                    X = np.concatenate((image_batch, generated_images))
                    y = np.concatenate((np.zeros((self.batch_size, 1, 64, 64)), 
                        np.ones((self.batch_size, 1, 64, 64))))
                    
                    d_loss = discriminator.train_on_batch(X, y)
                    print("batch %d d_loss : %f" % (index, d_loss))

                    discriminator.trainable = False

                    # Training C:
                    c_loss = classifier.evaluate(generated_images, label_batch)
                    #c_loss = classifier.train_on_batch(generated_images, label_batch)
                    print("batch %d c_loss : %f acc : %f" % (index, c_loss[0], c_loss[1]))

                    classifier.trainable = False
                    # Train G:
                    if self.temporal == False:
                        g_loss = discriminator_and_classifier_on_generator.train_on_batch(
                            [source_image_batch, noise], 
                            [image_batch, np.ones((self.batch_size, 1, 64, 64)), label_batch])
                    else:
                        g_loss = discriminator_and_classifier_on_generator.train_on_batch(
                            [noise], 
                            [image_batch, np.ones((100, 1, 64, 64)), label_batch])

                    discriminator.trainable = True
                    classifier.trainable = True
                    print("batch %d g_loss : %f" % (index, g_loss[1]))

                    
                    if index % 2000 == 0:
                        
                        loss_total.append((g_loss, d_loss, c_loss[0], c_loss[1]))
                        
                        generator.save_weights(self.models_path+
                            'gen_tmp_dacssGANs_'
                            + self.db+ "_"\
                            +self.mod
                            +"_"
                            +self.exp_tp, True)

                        discriminator.save_weights(self.models_path
                            +'discr_tmp_dacssGANs_'
                            + self.db+ "_"\
                            +self.mod 
                            +"_"
                            +self.exp_tp, True)

        loss_name = self.loss_path\
                    +"training_loss_for_GANS_"\
                    +self.mod\
                    + "_" + self.db\
                    +"_"+self.exp_tp+"_"\
                    + ".pkl" 

        self.hndl_obj.store_obj(loss_name, loss_total)


if __name__ == '__main__':
    dacssGANS_ = dacssGANs()
    dacssGANS_ .train()
