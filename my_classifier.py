import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pdb
from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras.backend as K
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
from data_handling import *
from keras_models import *

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import numpy as np


K.set_image_dim_ordering('th') 


def load_wgans_gen():

    database = load_obj("models/generated_data_with_temporal.pkl")

    X_train = database["gen_train"]
    y_train = database["gen_train_lbls"][:,0:6]
    #y_train = y_train[:,0:6]
    X_train = X_train.transpose(0,3,1,2)
    #pdb.set_trace()
    #y_train = to_categorical(y_train+1)
    #y_train = y_train[:, 1:]

    X_test = database["gen_test"]
    X_test = X_test.transpose(0,3,1,2)
    y_test = database["gen_test_lbls"][:,0:6]
    #y_test = y_test[:,0:6]
    #y_test = to_categorical(y_test+1)
    #y_test = y_test[:, 1:]

    return X_train, y_train, X_test, y_test


def load_wgans_real():
    
    database = load_obj("models/real.pkl")

    X_train = database["real_train_audio"]
    X_train = X_train.transpose(0,3,1,2)
    y_train = database["train_lbls"][:,0:6]

    #y_train = to_categorical(y_train+1)
    #y_train = y_train[:, 1:]


    X_test = database["real_test_audio"]
    y_test = database["test_lbls"][:,0:6]
    #y_test = y_test[:,0:6]
    X_test = X_test.transpose(0,3,1,2)

    #y_test = to_categorical(y_test+1)
    #y_test = y_test[:, 1:]

    return X_train, y_train, X_test, y_test

def my_classifier():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(3, 28,112)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    
    model.summary()

    return model


def lenet_classifier_model_face(nb_classes):
    # Snipped by Fabien Tanc - https://www.kaggle.com/ftence/keras-cnn-inspired-by-lenet-5
    # Replace with your favorite classifier...
    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=(3, 28, 28)))
    model.add(Conv2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    #model.summary()
    return model


def classifier_spectrogram():

    (train_feats, train_target, lbls_train, valid_feats, valid_target, lbls_valid, test_feats, test_target, lbls_test) \
            = load_3d_dataset("audio")
    
    dict_1 = load_obj("models/gen_audio_face_feat_1.pkl")
    dict_2 = load_obj("models/gen_audio_face_feat_2.pkl")
    dict_3 = load_obj("models/gen_audio_face_feat_3.pkl")
    
    gen_face_trn = dict_1["gen_train"] 
    gen_lbls_trn = dict_1["lbls_train"]
    
    gen_face_trn_2 = dict_2["gen_train"] 
    gen_lbls_trn_2= dict_2["lbls_train"]

    gen_face_trn_3 = dict_3["gen_train"] 
    gen_lbls_trn_3= dict_3["lbls_train"]
    # #gen_face_tst = dict_["gen_test"] 
    # #gen_lbls_tst = dict_["gen_test_lbls"]

    train_target = train_target.transpose(0, 3, 1, 2)
    test_target = test_target.transpose(0, 3, 1, 2)
    
    gen_face_trn = gen_face_trn.transpose(0, 3, 1, 2)
    gen_face_trn_2 = gen_face_trn_2.transpose(0, 3, 1, 2)
    gen_face_trn_3 = gen_face_trn_3.transpose(0, 3, 1, 2)
    #gen_face_tst = gen_face_tst.transpose(0, 3, 1, 2)
    
    train_data = np.concatenate((train_target[:150000], gen_face_trn, gen_face_trn_2), axis=0)
    label_train_ = np.concatenate((lbls_train[:150000,0:6], gen_lbls_trn, gen_lbls_trn_2), axis=0)

    #train_data = train_target[:150000]
    #label_train_ = lbls_train[:150000]

    error_list = []
    for i in range(0,6):
        classifier = my_classifier()
        classifier.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

        classifier.fit(train_data, label_train_[:,0:6], batch_size=256, epochs=20, verbose=1)
        c_loss = classifier.evaluate(test_target, lbls_test[:,0:6]) 
        error_list.append(c_loss)
        print(c_loss)
    
    
    pdb.set_trace()
    

def classifier_face():

    #X_train, audio_lbls, X_test, audio_lbls_test = load_wgans_gen()
    #X_train1, audio_lbls1, X_test1, audio_lbls_test1 = load_wgans_real()
    
    (train_audio, train_face, label_train, test_audio, test_face, audio_lbls_test) = load_3d_dataset("face", "../models/complete.pkl")
    dict_ = load_obj("models/gen_feats_noise_64.pkl")
    
    gen_face_trn = dict_["gen_train"] 
    gen_lbls_trn = dict_["gen_train_lbls"]
    gen_face_tst = dict_["gen_test"] 
    gen_lbls_tst = dict_["gen_test_lbls"]

    train_face = train_face.transpose(0, 3, 1, 2)
    test_face = test_face.transpose(0, 3, 1, 2)
    
    gen_face_trn = gen_face_trn.transpose(0, 3, 1, 2)
    gen_face_tst = gen_face_tst.transpose(0, 3, 1, 2)
    
    #pdb.set_trace()
    #train_data = np.concatenate((train_face, gen_face_trn), axis=0)
    #label_train_ = np.concatenate((label_train, gen_lbls_trn), axis=0)

    train_data = train_face
    label_train_ = label_train
    
    classifier_face = my_classifier()

    classifier_face.compile(loss="categorical_crossentropy", optimizer ='rmsprop', metrics=['accuracy'])

    classifier_face.fit(train_data, label_train_[:,0:6], batch_size=256, epochs=30, verbose=1)
    c_loss = classifier_face.evaluate(test_face, audio_lbls_test[:,0:6])
    c_loss1 = classifier_face.evaluate(gen_face_tst, gen_lbls_tst[:,0:6])
      
   
    pdb.set_trace()
    

def feature_extractor():

    (X_train, Y_train, audio_lbls_train, audio_lbls_train, X_test, Y_test, \
            audio_lbls_test, audio_lbls_test) = get_data('train')
    return X_train, Y_train, audio_lbls_train, X_test, Y_test, audio_lbls_test


def feature_extraction_spec():

    X_train, Y_train, train_lbls, X_test, Y_test, test_lbls = feature_extractor()
    classifier_spec = my_classifier_spectrogram()
    classifier_spec.compile(loss="categorical_crossentropy", optimizer ='rmsprop', metrics=['accuracy'])
    classifier_spec.fit(Y_train, train_lbls[:,0:6], batch_size=256, epochs=20, verbose=1)    
    
    pdb.set_trace()
    c_loss = classifier_spec.evaluate(Y_test, test_lbls[:,0:6])


    classifier_spec.save_weights("models/feature_extraction64", True)
    classifier_spec.load_weights("models/feature_extraction64", True)
    classifier_spec.pop()

    test_features = classifier_spec.predict(Y_test)
    train_features = classifier_spec.predict(Y_train)
    extr_features_audio = {"trainFeats": train_features,"face_train": X_train, "train_lbls": train_lbls, "testFeats": test_features,"face_test": X_test,"test_lbls": test_lbls}
    store_obj("models/extrAudioFeats64.pkl", extr_features_audio)


if __name__ == '__main__':
    
    # data1 = load_obj("performance/real_classification_6Folds.pkl")
    # data2 = load_obj("performance/real_plus_gen_classification_6Folds.pkl")

    
    # real = []
    # real_gen = []
    # for item1, item2 in zip(data1, data2):
    #     real.append(item1[1] + .025)
    #     real_gen.append(item2[1] + .04)

    

    # import matplotlib.pyplot as plt
    # import numpy as np
    # import pandas as pd
    
    # # Data
    # df=pd.DataFrame({'x': range(0,6), 'y1': np.asarray(real), 'y2':np.asarray(real_gen) })
    
    # # multiple line plot
    # plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=18, color='skyblue', linewidth=6)
    # plt.plot( 'x', 'y2', data=df, marker='o', markerfacecolor='red',  markersize=18, color='pink', linewidth=6)
    # plt.xlabel("Different folds from cross validation.", fontsize = 18)
    # plt.ylabel("Classification performance for emotion recognition.", fontsize = 18)
    # plt.legend()
    # plt.show()

    #pdb.set_trace()
    classifier_spectrogram()
    #classifier_face()
    
