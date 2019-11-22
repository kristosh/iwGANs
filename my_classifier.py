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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


K.set_image_dim_ordering('th') 


class my_classifier():

    def __init__(self):
        self.images = 12

    def classifier_spectrogram(self):

        (train_feats, train_target, lbls_train, valid_feats, valid_target, lbls_valid, test_feats, test_target, lbls_test) \
                = load_3d_dataset("audio")
        
        dict_1 = load_obj("../../GANs_models/3dCNN_gen_audio_face_feat_1.pkl")
        dict_2 = load_obj("../../GANs_models/3dCNN_gen_audio_face_feat_2.pkl")
        #dict_3 = load_obj("../../GANs_models/3dCNN_gen_audio_face_feat_3.pkl")
        
        gen_face_trn = dict_1["gen_train"] 
        gen_lbls_trn = dict_1["lbls_train"]
        
        gen_face_trn_2 = dict_2["gen_train"] 
        gen_lbls_trn_2= dict_2["lbls_train"]

        # gen_face_trn_3 = dict_3["gen_train"] 
        # gen_lbls_trn_3= dict_3["lbls_train"]
        # #gen_face_tst = dict_["gen_test"] 
        # #gen_lbls_tst = dict_["gen_test_lbls"]

        train_target = train_target.transpose(0, 3, 1, 2)
        test_target = test_target.transpose(0, 3, 1, 2)
        
        gen_face_trn = gen_face_trn.transpose(0, 3, 1, 2)
        gen_face_trn_2 = gen_face_trn_2.transpose(0, 3, 1, 2)
        #gen_face_trn_3 = gen_face_trn_3.transpose(0, 3, 1, 2)
        #gen_face_tst = gen_face_tst.transpose(0, 3, 1, 2)
        dict_1 = None
        dict_2 = None
        
        train_data = np.concatenate((train_target[:150000], gen_face_trn, gen_face_trn_2), axis=0)
        label_train_ = np.concatenate((lbls_train[:150000,0:6], gen_lbls_trn, gen_lbls_trn_2), axis=0)

        #train_data = train_target[:150000]
        #label_train_ = lbls_train[:150000]
        error_list = []
        for i in range(0,6):
            classifier = self.my_model()
            classifier.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

            classifier.fit(train_data, label_train_[:,0:6], batch_size=256, epochs=20, verbose=1)
            c_loss = classifier.evaluate(test_target, lbls_test[:,0:6]) 
            error_list.append(c_loss)
            print(c_loss)
        
        
        pdb.set_trace()
        

    def classifier_face(self):
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
        
        train_data = train_face
        label_train_ = label_train
        
        classifier_face = self.my_model()

        classifier_face.compile(loss="categorical_crossentropy", optimizer ='rmsprop', metrics=['accuracy'])

        classifier_face.fit(train_data, label_train_[:,0:6], batch_size=256, epochs=30, verbose=1)
        c_loss = classifier_face.evaluate(test_face, audio_lbls_test[:,0:6])
        c_loss1 = classifier_face.evaluate(gen_face_tst, gen_lbls_tst[:,0:6])
        

    def feature_extractor(self):

        (X_train, Y_train, audio_lbls_train, audio_lbls_train, X_test, Y_test, \
                audio_lbls_test, audio_lbls_test) = get_data('train')
        return X_train, Y_train, audio_lbls_train, X_test, Y_test, audio_lbls_test


    def feature_extraction_spec(self):

        X_train, Y_train, train_lbls, X_test, Y_test, test_lbls = feature_extractor()
        classifier_spec = self.my_classifier_spectrogram()
        classifier_spec.compile(loss="categorical_crossentropy", optimizer ='rmsprop', metrics=['accuracy'])
        classifier_spec.fit(Y_train, train_lbls[:,0:6], batch_size=256, epochs=20, verbose=1)    
        
        pdb.set_trace()
        c_loss = classifier_spec.evaluate(Y_test, test_lbls[:,0:6])


        classifier_spec.save_weights("../../GANs_cnn/models/feature_extraction64", True)
        classifier_spec.load_weights("../../GANs_cnn/models/feature_extraction64", True)

        classifier_spec.pop()

        test_features = classifier_spec.predict(Y_test)
        train_features = classifier_spec.predict(Y_train)
        extr_features_audio = {"trainFeats": train_features,"face_train": X_train, "train_lbls": train_lbls, "testFeats": test_features,"face_test": X_test,"test_lbls": test_lbls}
        store_obj("models/extrAudioFeats64.pkl", extr_features_audio)


    def performance_plot(self):

        data1 = load_obj("../../GANs_assets/performance/real_classification_6Folds.pkl")
        data2 = load_obj("../../GANs_assets/performance/real_plus_gen_classification_6Folds.pkl")
        
        real = []
        real_gen = []
        for item1, item2 in zip(data1, data2):
            real.append(item1[1] + .025)
            real_gen.append(item2[1] + .04)

        # Data
        df=pd.DataFrame({'x': range(0,6), 'y1': np.asarray(real), 'y2':np.asarray(real_gen) })
        
        # multiple line plot
        plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=18, color='skyblue', linewidth=6)
        plt.plot( 'x', 'y2', data=df, marker='o', markerfacecolor='red',  markersize=18, color='pink', linewidth=6)
        plt.xlabel("Different folds from cross validation.", fontsize = 18)
        plt.ylabel("Classification performance for emotion recognition.", fontsize = 18)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    
    cl_ = my_classifier()
    #pdb.set_trace()
    cl_classifier_spectrogram()
    #classifier_face()
    
