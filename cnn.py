from __future__ import print_function

import os
def cls(): os.system('clear')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

nClasses = 6
epochs = 20


def run_classifier():


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

    train_target = train_target.transpose(0, 3, 1, 2)
    test_target = test_target.transpose(0, 3, 1, 2)
    
    gen_face_trn = gen_face_trn.transpose(0, 3, 1, 2)
    gen_face_trn_2 = gen_face_trn_2.transpose(0, 3, 1, 2)
    gen_face_trn_3 = gen_face_trn_3.transpose(0, 3, 1, 2)

    train_data = np.concatenate((train_target[:150000], gen_face_trn, gen_face_trn_2, gen_face_trn_3), axis=0)
    train_labels = np.concatenate((lbls_train[:150000,0:6], gen_lbls_trn, gen_lbls_trn_2, gen_lbls_trn_3), axis=0)

    test_data = test_target 
    test_labels = lbls_test[:,0:6]

    pdb.set_trace()
    # Find the unique numbers from the train labels
    classes = 6
    
    nClasses = 6
    print('Total number of outputs : ', nClasses)
    print('Output classes : ', classes)

    pdb.set_trace()
    # Find the shape of input images and create the variable input_shape
    nDims, nRows,nCols = train_data.shape[1:]

    input_shape = (nDims, nRows, nCols)

    # Change to float datatype
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    # train_images_cl = train_source_data.astype('float32')
  
    batch_size = 128
    
    model1 = cnnModel(input_shape, nClasses)
    model1.summary()
    
    c_optim = RMSprop(lr=0.0001, decay=1e-6)

    total_loss = []
    total_cm = []
    pdb.set_trace()
    
    for iteration in range(0,6):

        model1.compile(optimizer=c_optim, loss='categorical_crossentropy', metrics=['accuracy'])
        history = model1.fit(train_data, train_labels, batch_size=batch_size, epochs=30, verbose=1, validation_data=(train_data, train_labels))
        
        c_loss = model1.evaluate(test_data, lbls_test[:,0:6]) 
        y_pred = model1.predict(test_data)
        
        ret = (y_pred.argmax(axis=1)+1)
        test_labels_ = (test_labels.argmax(axis=1)+1)
        cnf_matrix = confusion_matrix(test_labels_, ret)

        
        class_names = ["hap", "sad", "ang", "fea", "dis", "neu"]
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix for generated specs')
        plt.savefig("confusionMatrix_genPlusReal_"+str(iteration)+".png")
        plt.close()
        total_loss.append(c_loss)
        total_cm.append(cnf_matrix)

    pdb.set_trace()
    store_obj("total_loss.pkl", total_loss)
    store_obj("total_cm.pkl", total_cm)

def run_ind_conformal_prefiction():

    real_train_data, train_faces, train_lbls, real_test_data, test_faces, test_lbls, \
        real_c_data, c_faces, c_lbls = load_real_samples_RAV()

    inductive_conformal_prediction(train_faces, train_lbls, test_faces, test_lbls, c_faces, c_lbls, \
        real_train_data, train_lbls, real_test_data, test_lbls)

    # inductive_conformal_prediction(train_data, train_labels_one_hot, test_data, test_labels_one_hot, train_images_cl, train_labels_cl_one_hot, \
    #     test_target_data, test_target_lbls_one_hot, vld_target_data, vld_target_lbls_one_hot, training_target_data, training_target_lbls_one_hot)


if __name__ == '__main__':
    run_classifier()
    #run_ind_conformal_prefiction()

