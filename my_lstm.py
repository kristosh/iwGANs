# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
import keras
import pdb
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from data_handling import *
import pdb


def load_my_dataset(fold_index):

    temp_path = '../../GANs_models/tmp_dtst/complete_'+str(fold_index)+'.pkl'
    my_3d_dict = load_obj(temp_path)

    train_feats = my_3d_dict["feats_train"]
    lbls_train = my_3d_dict["lbls_train"]
    target_train = my_3d_dict["target_train"]

    target_train = (target_train.astype(np.float32) - 127.5)/127.5

    temp_path = '../../GANs_models/tmp_dtst/complete_7.pkl'
    my_3d_dict = load_obj(temp_path)

    test_feats = my_3d_dict["feats_train"]
    lbls_test = my_3d_dict["lbls_train"]
    target_test = my_3d_dict["target_train"]

    target_test= (target_test.astype(np.float32) - 127.5)/127.5

    return train_feats, lbls_train, target_train, test_feats, lbls_test, target_test


def lstm_model():
    
    model = Sequential()
    model.add(LSTM(units = 256, input_shape = (10, 256), return_sequences = False, dropout = 0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(6, activation='sigmoid'))

    print(model.summary())
    return model


def lstm_main():

    tensorboard = keras.callbacks.TensorBoard(
            log_dir='my_tf_logs/lstm',
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=True
        )

    model = lstm_model()
    # model = Sequential()
    # model.add(LSTM(units = 256, input_shape = (10, 256), return_sequences = False, dropout = 0.5))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(64, activation='sigmoid'))
    # model.add(Dense(6, activation='sigmoid'))

    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    

    tmp_ = load_obj('../../GANs_models/tmp_dtst/lstm_1.pkl')
    
    feats_train = tmp_['feats_train']
    lbls_train = tmp_['lbls_train']
    target_train = tmp_['target_train']

    pdb.set_trace()
    for fold_indx in range(1, 6):
        train_feats, lbls_train, target_train, test_feats, lbls_test, target_test = load_my_dataset(fold_indx)
        model.fit(train_feats, lbls_train, epochs=15, batch_size=64)
        tensorboard.set_model(model)
        pred_feats_test = model.predict(test_feats)
        c_loss = model.evaluate(test_feats, lbls_test) 
        print (c_loss)

    pdb.set_trace()
    model.save_weights("../../GANs_models/lstm_weights")
    model.load_weights("../../GANs_models/lstm_weights")
    
    model.pop()
    #pdb.set_trace()
   
    for fold_indx in range(1, 8):

        train_feats, lbls_train, target_train, test_feats, lbls_test, target_test = load_my_dataset(fold_indx)
        pred_feats_train = model.predict(train_feats)
        pred_feats_test= model.predict(test_feats)
        dataset_dict1 = {"feats_train":pred_feats_train,"lbls_train": lbls_train, "target_train": target_train}
        store_obj("../../GANs_models/tmp_dtst/lstm_"+str(fold_indx)+".pkl", dataset_dict1)

    pdb.set_trace()


if __name__ == "__main__":
    lstm_main()
    