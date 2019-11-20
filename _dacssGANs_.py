import numpy as np
import glob, pickle
import os, sys

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
from data_handling import *
from keras_models import *


img_rows = 28
img_cols = 28
in_ch = 3
batch_size = 100
source_modality = "face"
target_modality = "audio"
temporal = True


def generator_containing_discriminator_and_classifier(generator, discriminator, classifier, temporal):
    
    if temporal == False:
        inputs = Input((in_ch, img_cols, img_rows))
        input_conditional = Input(shape=(13,))
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
        model = Model(input=[inputs, input_conditional], output=[x_generator, x_discriminator, x_classifier])
    else:
        model = Model(input=[inputs], output=[x_generator, x_discriminator, x_classifier])

    return model


def combine_images(generated_images):

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


def discriminator_loss(y_true,y_pred):
    batch_size=100
    return K.mean(K.binary_crossentropy(K.flatten(y_pred), K.concatenate([K.zeros_like(K.flatten(y_pred[:batch_size,:,:,:])),K.ones_like(K.flatten(y_pred[:batch_size,:,:,:])) ]) ), axis=-1)


def discriminator_on_generator_loss(y_true,y_pred):
    batch_size=100
    return K.mean(K.binary_crossentropy(K.flatten(y_pred), K.ones_like(K.flatten(y_pred))), axis=-1)


def generator_l1_loss(y_true,y_pred):
    batch_size=100
    return K.mean(K.abs(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)


def train(batch_size): 

    if temporal == True:
        (X_train, Y_train, label_train, audio_lbls_train, X_test, Y_test, 
            audio_lbls_test, audio_lbls_test, semantics_train, semantics_test) = load_3d_dataset()
    else:
        (X_train, Y_train, label_train, audio_lbls_train, X_test, Y_test, 
            audio_lbls_test, audio_lbls_test, semantics_train, semantics_test) = get_data('train')

    Y_train = Y_train.transpose(0, 3, 1, 2)

    discriminator = discriminator_model()
    if temporal == False:
        generator = generator_model()
    else:
        generator = generator_model_temporal()

    classifier = lenet_classifier_model(6)

    discriminator_and_classifier_on_generator = generator_containing_discriminator_and_classifier(
        generator, discriminator, classifier, temporal)

    d_optim = Adagrad(lr=0.00001)
    g_optim = Adagrad(lr=0.00001)
    
    generator.compile(loss=generator_l1_loss, optimizer=g_optim)

    discriminator_and_classifier_on_generator.compile(
        loss=[generator_l1_loss, discriminator_on_generator_loss, "categorical_crossentropy"],
        optimizer="rmsprop")

    discriminator.trainable = True
    discriminator.compile(loss=discriminator_loss, optimizer=d_optim) # rmsprop
    classifier.trainable = True
    c_optim = Adagrad(lr=0.00001)
    classifier.compile(loss="categorical_crossentropy", optimizer=c_optim, metrics=['accuracy'])
 
    #generator.load_weights('temp_files/generator_c_audio')
    #discriminator.load_weights('temp_files/discriminator_c_audio')
    classifier.load_weights('models/classifier_28x112_'+target_modality)
    #generated_data_dacsGANs(X_train, X_test, audio_lbls_train, audio_lbls_test, Y_test, generator)
    #pdb.set_trace()

    for epoch in range(280):
        
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / batch_size))
        for index in range(int(X_train.shape[0] / batch_size)):
            
            source_image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            image_batch = Y_train[index * batch_size:(index + 1) * batch_size]
            label_batch = label_train[index * batch_size:(index + 1) * batch_size]  # replace with your data here

            #pdb.set_trace()
            noise = np.random.normal(0, 1, (batch_size, 100))
            noise = np.concatenate([noise, label_batch[:,0:6], source_image_batch], axis = 1)

            if temporal == False:
                generated_images = generator.predict([X_train[index * batch_size:(index + 1) * batch_size], label_batch])
            else:
                generated_images = generator.predict(noise)
            
            num, row,col,ch= image_batch.shape
            mean = 0
            var = 0.1
            sigma = var**0.5

            gauss1 = np.random.normal(mean,sigma,(num, row,col,ch))
            gauss1 = gauss1.reshape(num, row,col,ch)
            gauss1 = 0.01*gauss1
            gauss1 = gauss1 +np.amax(gauss1)
            image_batch = image_batch + gauss1

            gauss2 = np.random.normal(mean,sigma,(num, row,col,ch))
            gauss2 = gauss2.reshape(num, row,col,ch)
            gauss2 = 0.01*gauss2
            gauss2 = gauss2 +np.amax(gauss2)
            
            generated_images = generated_images +gauss2

            if index % 200 == 0:       
                file_name = "generated_imgs/"+target_modality+"/generated"+str(epoch)+"_"+str(index)+".png"
                store_image_maps(generated_images, file_name)               
                file_name = "generated_imgs/"+target_modality+"/target_"+str(epoch)+"_"+str(index)+".png"
                store_image_maps(image_batch, file_name)

            # Training D:
            X = np.concatenate((image_batch, generated_images))
            y = np.concatenate((np.zeros((100, 1, 64, 64)), np.ones((100, 1, 64, 64))))
            #y = np.zeros((20, 1, 64, 64))  # [1] * batch_size + [0] * batch_size
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))

            discriminator.trainable = False

            #pdb.set_trace() 
            # Training C:
            # c_loss = classifier.train_on_batch(generated_images, label_batch[:,0:6])
            c_loss = classifier.evaluate(generated_images, label_batch[:,0:6])
            print("batch %d c_loss : %f acc : %f" % (index, c_loss[0], c_loss[1]))

            classifier.trainable = False
            # Train G:
            #@pdb.set_trace()
            if temporal == False:
                g_loss = discriminator_and_classifier_on_generator.train_on_batch([X_train[index * batch_size:(index + 1) * batch_size, :, :, :], label_batch], 
                    [image_batch, np.ones((100, 1, 64, 64)), label_batch[:,0:6]])
            else:
                g_loss = discriminator_and_classifier_on_generator.train_on_batch([noise], [image_batch, np.ones((100, 1, 64, 64)), label_batch[:,0:6]])


            discriminator.trainable = True
            classifier.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss[1]))
            if index % 20 == 0:
                generator.save_weights('models/gen_tmp_dacssGANs_'+target_modality, True)
                discriminator.save_weights('models/discr_tmp_dacssGANs_'+target_modality, True)


if __name__ == '__main__':
    #load_temp()
    train(100)
