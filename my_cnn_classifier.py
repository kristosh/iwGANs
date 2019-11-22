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
K.set_image_dim_ordering('th')

from data_handling import *
from keras_models import *


from data_handling import *


class my_cnn_classifier():


    def __init__(self):

        self.no_of_layers = 32
        self.batch_size = 128
        self.nClasses = 6
        self.c_optim = RMSprop(lr=0.0001, decay=1e-6)
        self.epochs = 15


    def load_gen_data(self):

        data_list = []
        lbls_list = []
        for index in range(1, 4):      
            dict_ = load_obj("../../GANs_models/3dCNN_gen_audio_face_feat_"+str(index)+".pkl")            
            data = dict_["gen_train"] 
            lbls_list.append(dict_["lbls_train"])       
            data_list.append(data.transpose(0, 3, 1, 2))
        
        return data_list[0], data_list[1], data_list[2], lbls_list[0], lbls_list[1], lbls_list[2]


    def run_classifier(self):

        (train_feats, train_target, lbls_train, valid_feats, valid_target, lbls_valid, test_feats, test_target, lbls_test) \
                = load_3d_dataset("audio")

        train_target = train_target.transpose(0, 3, 1, 2)[:150000]
        test_target = test_target.transpose(0, 3, 1, 2)

        gen_1, gen_2, gen_3, gen_lbls_1, gen_lbls_2, gen_lbls_3 = self.load_gen_data()

        #train_data = np.concatenate((train_target,  gen_1, gen_2), axis=0)
        #train_labels = np.concatenate((lbls_train[:150000,0:6], gen_lbls_1, gen_lbls_2), axis=0)

        test_data = test_target 
        test_labels = lbls_test[:,0:6]

        train_data = train_target
        train_labels = lbls_train[:150000,0:6]
        
        del train_feats
        del train_target
        del valid_target
        del test_target
        del gen_1, gen_2, gen_3

        pdb.set_trace()
        # Find the unique numbers from the train labels  
        print('Total number of outputs : ', self.nClasses)
        # Find the shape of input images and create the variable input_shape
        nDims, nRows,nCols = train_data.shape[1:]
        input_shape = (nDims, nRows, nCols)

        # Change to float datatype
        train_data = train_data.astype('float32')
        test_data = test_data.astype('float32')
        
        model = cnnModel(input_shape, self.nClasses)
        model.summary()
        
        total_loss = []
        total_cm = []
        pdb.set_trace()
        
        for iteration in range(0,6):
            model.compile(optimizer=self.c_optim, loss='categorical_crossentropy', metrics=['accuracy'])
            history = model.fit(train_data, train_labels, batch_size=self.batch_size, epochs = self.epochs, verbose=1, validation_data=(train_data, train_labels))      
            
            c_loss = model.evaluate(test_data, lbls_test[:,0:6]) 
            print (c_loss)
            # y_pred = model.predict(test_data)
            # total_loss.append(c_loss)      
            # cnf_matrix = self.plot_confusion_matrix(y_pred, lbls_test[:,0:6], iteration)
            # total_cm.append(cnf_matrix)
            # pdb.set_trace()

        # store_obj("total_loss.pkl", total_loss)
        # store_obj("total_cm.pkl", total_cm)


    def plot_confusion_matrix(self, y_pred, test_labels, iteration):

        ret = (y_pred.argmax(axis=1)+1)
        test_labels_ = (test_labels.argmax(axis=1)+1)
        cnf_matrix = confusion_matrix(test_labels_, ret)   
        class_names = ["hap", "sad", "ang", "fea", "dis", "neu"]
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix for generated specs')
        plt.savefig("confusionMatrix_genPlusReal_"+str(iteration)+".png")
        plt.close()


if __name__ == '__main__':
    my_cnn = my_cnn_classifier()
    my_cnn.run_classifier()
    #run_ind_conformal_prefiction()

