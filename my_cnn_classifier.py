from __future__ import print_function

import os
def cls(): os.system('clear')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import matplotlib
matplotlib.use('Agg')

#import pylab as plt
import keras
import cv2
import keras.models as km
from sklearn.utils import shuffle
from keras.optimizers import SGD, Adagrad, RMSprop

import pdb
from sklearn.metrics import confusion_matrix
from confusion_matrix import plot_confusion_matrix
from keras.utils import to_categorical

from keras import backend as K

K.common.set_image_dim_ordering('th')


from data_handling import data_handle
from keras_models import *
import matplotlib.pyplot as plt

from data_handling import *


class my_cnn_classifier():


    def __init__(self):

        self.no_of_layers = 32
        self.batch_size = 128
        self.nClasses = 6
        self.c_optim = RMSprop(lr=0.0001, decay=1e-6)
        self.epochs = 20

        self.type_of_features = "3dCNN"
        self.db = "cremad"
        self.target_mod = "face"

        self.my_obj = data_handle()

        self.version = "both"
        self.type_of_input = "with_source"
        
        self.cM_name = "../assets/cm_"+ \
            self.type_of_features +"_" + \
            self.db +"_" + \
            self.target_mod +"_" + \
            self.version +"_" + \
            self.type_of_input


    def generated_faces (self, type_of_feats):

        path = "../../GANs_models/cremad/"
        _gFile_ = "3dCNN_creamad_generated_"+self.type_of_input+".pkl"

        _dct_ = self.my_obj.load_obj(path + _gFile_)

        return _dct_


    def load_gen_data(self, type_of_feats):

        #"3dCNN_ravdess_gen_audio_face_feat__3dCNN_0"
        data_list = []
        lbls_list = []
        for index in range(0, 2):      
            #pdb.set_trace()
            dict_ = self.my_obj.load_obj("../../GANs_models/"+
                type_of_feats+
                "_gen_audio_face_feat_"+
                str(index)+".pkl")            

            #dict_ = self.my_obj.load_obj("../../GANs_models/"+type_of_feats+"_"+self.db+"_gen_audio_face_feat_"+str(index)+".pkl")            
            
            data = dict_["gen_train"] 
            lbls_list.append(dict_["lbls_train"])       
            data_list.append(data.transpose(0, 3, 1, 2))
        
        return data_list[0], data_list[1], lbls_list[0], lbls_list[1]


    def run_classifier(self):

        if self.target_mod == "face":
           
            _dct_  = self.my_obj.load_3d_dataset(self.target_mod)

            real_train = _dct_["trn_trg"].transpose(0, 3, 1, 2)

            test_data = _dct_["tst_trg"].transpose(0, 3, 1, 2)
            test_labels = _dct_["tst_lbls"]

            _dctG_ = self.generated_faces(self.target_mod)

            gen_train = _dctG_["gen_train"].transpose(0, 3, 1, 2)
            gen_lbls = _dctG_["lbls_train"]

            if self.version == "real":
                train_data = real_train 
                train_labels = _dct_["trn_lbls"]
            elif self.version == "generated":
                train_data = gen_train 
                train_labels = gen_lbls 
            else:
                train_data = np.concatenate((real_train,  gen_train), axis=0)
                train_labels =  np.concatenate((_dct_["trn_lbls"],  gen_lbls), axis=0)
            del _dct_, _dctG_
        else:
            #_dct_  = self.my_obj.load_3d_dataset_rav("audio")
            train_target = _dct_["trn_fts"].transpose(0, 3, 1, 2)
            test_target = _dct_["tst_fts"].transpose(0, 3, 1, 2)
            #gen_1, gen_2, gen_lbls_1, gen_lbls_2 = self.load_gen_data(self.type_of_features)
            #train_target = train_target[:(gen_1.shape[0]+ gen_1.shape[1])] #.train_data
            #train_labels = _dct_["trn_lbls"][:(gen_1.shape[0]+ gen_1.shape[1])]
            #train_data = np.concatenate((train_target,  gen_1), axis=0)
            #train_labels = np.concatenate((train_labels, gen_lbls_1), axis=0)    
            train_data = train_target
            train_labels = _dct_["trn_lbls"]
            #train_labels = _dct_["trn_lbls"][:(gen_1.shape[0]+ gen_1.shape[1])]
            test_data = test_target 
            test_labels = _dct_["tst_lbls"]
            del _dct_
            del gen_1, gen_2

        # Find the unique numbers from the train labels  
        print('Total number of outputs : ', self.nClasses)
        # Find the shape of input images and create the variable input_shape
        nDims, nRows,nCols = train_data.shape[1:]
        input_shape = (nDims, nRows, nCols)

        # Change to float datatype
        train_data = train_data.astype('float32')
        test_data = test_data.astype('float32')
        
        if self.target_mod == "audio":
            model = cnnModel(input_shape, self.nClasses)
            model.summary()
            
            total_loss = []
            total_cm = []
            
            for iteration in range(0,4):
                model.compile(optimizer=self.c_optim, loss='categorical_crossentropy', metrics=['accuracy'])
                history = model.fit(train_data, train_labels, 
                    batch_size=self.batch_size, 
                    epochs = self.epochs, 
                    verbose=1, 
                    validation_data=(train_data, train_labels))      
                
                c_loss = model.evaluate(test_data, test_labels) 
                print (c_loss)
                y_pred = model.predict(test_data)
                total_loss.append(c_loss)  

                cnf_matrix = self._conf_matrix_(y_pred, 
                    test_labels,  
                    self.cM_name+".png")

                total_cm.append(cnf_matrix)
            
            pdb.set_trace()

            self.my_obj.store_obj("total_loss_transfoermers_real.pkl", total_loss)
            self.my_obj.store_obj("total_cm_transormers_real.pkl", total_cm)
            model.save_weights("spec_classifier")

        else:
            nDims, nRows,nCols = train_data.shape[1:]
            input_shape = (nDims, nRows, nCols)
            model = cnnModel(input_shape, self.nClasses)
            #model = lenet_classifier_model_face(self.nClasses)
            model.summary()
            
            total_loss = []
            total_cm = []
            for iteration in range(0,4):
                model.compile(optimizer=self.c_optim, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

                history = model.fit(train_data, train_labels, 
                    batch_size=self.batch_size, 
                    epochs = self.epochs, 
                    verbose=1, 
                    validation_data=(train_data, train_labels))      
                
                c_loss = model.evaluate(test_data, test_labels) 
                print (c_loss)
                y_pred = model.predict(test_data)
                total_loss.append(c_loss)      
                cnf_matrix = self._conf_matrix_(y_pred, 
                    test_labels,  
                    self.cM_name+".png")

                total_cm.append(cnf_matrix)

            loss_file = "../models/total_loss_"+ \
                self.target_mod+"_"+ \
                self.version+"_"+ \
                self.type_of_input+ \
                ".pkl"

            self.my_obj.store_obj(loss_file, total_loss)
            cm_file = "../models/total_cm_"+ \
                self.target_mod+"_"+ \
                self.version+"_"+ \
                self.type_of_input+ \
                ".pkl"


            self.my_obj.store_obj(cm_file, total_cm)
            pdb.set_trace()


    def _conf_matrix_(self, y_pred, test_labels, filename):

        ret = (y_pred.argmax(axis=1)+1)
        test_labels_ = (test_labels.argmax(axis=1)+1)
        cm = confusion_matrix(test_labels_, ret)   
        class_names = ["hap", "sad", "ang", "fea", "dis", "neu"]
        #plot_confusion_matrix(cm, filename, class_names)

        return cm
        
    
    def extract_features_form_spectrograms(self):

        _dct_  = self.my_obj.load_3d_dataset("face")

        train_target = _dct_["trn_fts"].transpose(0, 3, 1, 2)
        train_source = _dct_["trn_trg"]
        train_lbls = _dct_["trn_lbls"]

        test_target = _dct_["tst_fts"].transpose(0, 3, 1, 2)
        test_source = _dct_["tst_trg"]
        test_lbls = _dct_["tst_lbls"]

        nDims, nRows,nCols = train_target.shape[1:]
        input_shape = (nDims, nRows, nCols)

        model = cnnModel(input_shape, self.nClasses)
        model.compile(optimizer=self.c_optim, loss='categorical_crossentropy', metrics=['accuracy'])

        model.load_weights("../models/spec_classifier")
        
        model.pop()

        test_target = model.predict(test_target)
        train_target = model.predict(train_target)

        _str_dct = {"trn_fts": train_target, 
            "trn_trg": train_source, 
            "trn_lbls": train_lbls, 
            "tst_fts": test_target,
            "tst_trg": test_source,
            "tst_lbls": test_lbls
            }

        self.my_obj.store_obj("2dCNN_features.pkl", _str_dct)
        pdb.set_trace()
        

if __name__ == '__main__':
    my_cnn = my_cnn_classifier()
    my_cnn.run_classifier()
    #run_ind_conformal_prefiction()
    #my_cnn.extract_features_form_spectrograms()

