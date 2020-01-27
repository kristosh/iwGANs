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
from keras_models import *
import pdb


class my_LSTM():

    def __init__(self):

        self.no_of_layers = 64

    def load_my_dataset_lstm(self, fold_index):

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

    def lstm_main(self):

        tensorboard = keras.callbacks.TensorBoard(
                log_dir='my_tf_logs/lstm',
                histogram_freq=0,
                batch_size=batch_size,
                write_graph=True,
                write_grads=True
            )

        model = lstm_model(self.no_of_layers)

        print(model.summary())
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        
        for fold_indx in range(1, 6):
             train_feats, lbls_train, target_train, test_feats, lbls_test, target_test = self.load_my_dataset_lstm(fold_indx)
             #pdb.set_trace()
             model.fit(train_feats, lbls_train, epochs=20, batch_size=64)
             #tensorboard.set_model(model)
             pred_feats_test = model.predict(test_feats)
             c_loss = model.evaluate(test_feats, lbls_test) 
             print (c_loss)
        

        pdb.set_trace()
        
        # model.save_weights("../../GANs_models/lstm_weights_"+str(self.no_of_layers))
        model.load_weights("../../GANs_models/lstm_weights_"+str(self.no_of_layers))

        model.pop()
        for fold_indx in range(1, 8):

            train_feats, lbls_train, target_train, test_feats, lbls_test, target_test = self.load_my_dataset_lstm(fold_indx)
            pred_feats_train = model.predict(train_feats)
            pred_feats_test= model.predict(test_feats)
            dataset_dict1 = {"feats_train":pred_feats_train,"lbls_train": lbls_train, "target_train": target_train}
            store_obj("../../GANs_models/tmp_dtst/lstm_"+str(self.no_of_layers)+"_"+str(fold_indx)+".pkl", dataset_dict1)

        pdb.set_trace()


if __name__ == "__main__":
    lstm = my_LSTM()
    lstm.lstm_main()
    