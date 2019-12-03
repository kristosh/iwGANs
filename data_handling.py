import pickle
import numpy as np
from keras.utils import to_categorical
import pdb
from keras_models import *
from keras.optimizers import SGD, Adagrad
import cv2
import os

source_modality = "face"
target_modality = "audio"


def load_obj(filename):
    with open(filename, "rb") as fp:  # Unpickling
        return pickle.load(fp, encoding='bytes') 


def store_obj(filename, obj):
    with open(filename, "wb") as fp:  # Pickling
        pickle.dump(obj, fp, protocol=4)


def get_data_cp(datadir):

    # face_classifier = lenet_classifier_model_face(6)
    # face_classifier.trainable = True
    # c_optim = Adagrad(lr=0.00001)
    # face_classifier.compile(loss="categorical_crossentropy", optimizer=c_optim, metrics=['accuracy'])
    # face_classifier.load_weights('temp_files/classifier_28x28_face', True)

    cnn_dict_train = load_obj("../models/_full_spec_new_train_CP_56_112.pkl")
    X_train = np.transpose(cnn_dict_train["face_data"] , (0, 3, 1, 2))
    Y_train = np.transpose(cnn_dict_train["audio_data"] , (0, 3, 1, 2))

    face_lbls_train = cnn_dict_train["face_lbls"]
    semantics_train = cnn_dict_train["semantics"]
    
    audio_lbls_train = cnn_dict_train["audio_lbls"]
    #audio_lbls_train = face_classifier.predict_classes(X_train) + 1
    face_lbls_train = audio_lbls_train

    # overlap = [i for i, j in zip(face_lbls_train, audio_lbls_train) if i == j]
    # first = float(len(overlap))
    # second = float((audio_lbls_train_.shape[0]))
    
    # print ((float(len(overlap))/float((audio_lbls_train_.shape[0])))*100)

    cnn_dict_test = load_obj("../models/_full_spec_new_validate_56_112.pkl")
    
    X_test = np.transpose(cnn_dict_test["face_data"], (0, 3, 1, 2))
    Y_test = np.transpose(cnn_dict_test["audio_data"], (0, 3, 1, 2))
    
    audio_lbls_test = cnn_dict_test["audio_lbls"]
    #audio_lbls_test= face_classifier.predict_classes(X_test) + 1
    
    face_lbls_test = cnn_dict_test["face_lbls"]
    semantics_test = cnn_dict_test["semantics"]

    audio_lbls_test = np.reshape(audio_lbls_test, [audio_lbls_test.shape[0], 1])
    audio_lbls_train = np.reshape(audio_lbls_train, [audio_lbls_train.shape[0], 1])

    face_lbls_test = np.reshape(face_lbls_test, [face_lbls_test.shape[0], 1])
    face_lbls_train = np.reshape(face_lbls_train, [face_lbls_train.shape[0], 1])
    #pdb.set_trace()
    return X_train, Y_train, face_lbls_train, audio_lbls_train, X_test, Y_test, face_lbls_test, audio_lbls_test, semantics_train, semantics_test


def load_temp():

    dict_cnn = load_obj("temp_files/cnn_dict_"+target_modality+".pkl")
    train = dict_cnn["training_data"]
    train_total_lbls = dict_cnn["training_lbls"]
    test = dict_cnn["test_gen"]
    test_lbls = dict_cnn["test_lbls"]

    np.random.seed(0)
    ind1 = np.random.choice(range(train.shape[0]), size=(37000,), replace=False)
    rest1 = np.array([i for i in range(0, train.shape[0]) if i not in ind1])
    rest1 = np.array(rest1)

    #pdb.set_trace()
    train_db = train[rest1, :, :, :]
    valid_db = train[ind1, :, :, :]
    train_lbls = train_total_lbls[rest1, :]
    valid_lbls = train_total_lbls[ind1, :]

    classifier_ = lenet_classifier_model(6)
    classifier_.trainable = True
    c_optim = Adagrad(lr=0.001)
    classifier_.compile(loss="categorical_crossentropy", optimizer=c_optim, metrics=['accuracy'])
    classifier_.fit(train_db, train_lbls[:,0:6], epochs=5, batch_size=256, verbose=1,  validation_data=(valid_db, valid_lbls[:,0:6]))
    c_loss = classifier_.evaluate(test, test_lbls[:,0:6])

    confusion_m(test, classifier_, test_lbls[:,0:6])
    #pdb.set_trace()

def load_real_samples_RAV():
    
    cnn_dict= load_obj("../models/rav_v4.pkl")

    train_face = cnn_dict["faces_Tr"]
    train_sp = cnn_dict["specs_Tr"]
    train_lbls = cnn_dict["ann_Tr"]

    test_face = cnn_dict["faces_C"]
    test_sp = cnn_dict["specs_C"]
    test_lbls = cnn_dict["ann_C"]

    train_sp = train_sp.transpose(0,3,1,2)
    test_sp = test_sp.transpose(0,3,1,2)

    train_face = train_face.transpose(0,3,1,2)
    test_face = test_face.transpose(0,3,1,2)

    train_lbls = train_lbls.astype(int)
    test_lbls = test_lbls.astype(int)

    train_lbls = to_categorical(train_lbls)
    train_lbls = train_lbls[:, 1:]
    train_lbls = np.concatenate((train_lbls, train_lbls, train_lbls),axis=1)

    test_lbls = to_categorical(test_lbls)
    test_lbls = test_lbls[:, 1:]
    test_lbls = np.concatenate((test_lbls,test_lbls, test_lbls), axis=1)

    train_face = (train_face.astype(np.float32) - 127.5)/127.5
    train_sp = (train_sp.astype(np.float32) - 127.5)/127.5

    test_face = (test_face.astype(np.float32) - 127.5)/127.5
    test_sp = (test_sp.astype(np.float32) - 127.5)/127.5

    return train_face, train_sp, train_lbls[:,:13], test_face, test_sp, test_lbls[:,:13]


def gen_data_RAV():

    #generator = generator_model_unet()
    generator = generator_model()
    g_optim = Adagrad(lr=0.0001)
    generator.compile(loss=generator_l1_loss, optimizer=g_optim)
    generator.load_weights('temp_files/dacssGAN_gen_specs_RAVDESS_with_lbls_audio')

    train_face, train_sp, train_lbls, test_face, test_sp, test_lbls = load_real_samples_RAV()

    (train_face, train_lbls) = shuffle(train_face_, train_lbls)
    
    train_face_ = train_face[1:10001, :, :, :]
    train_lbls_ = train_lbls[1:10001,:]
    #train_lbls_ = np.zeros((5000, 13))

    #zeros = -3*np.ones((5000, 3, 28, 28))
    #train_face_ =  zeros
    generated_train = generator.predict([train_face_, train_lbls_])
    generated_test = generator.predict([test_face, test_lbls])

    pdb.set_trace()

    gen_data_dictionary = {"gen_train": generated_train, "gen_train_lbls": train_lbls, "gen_test": generated_test, "gen_test_lbls": test_lbls, "real_train": train_sp, "real_train_lbls": train_lbls, "real_test": test_sp, "real_test_lbls": test_lbls}
    store_obj("temp_files/rav_data_v4.pkl", gen_data_dictionary)
    
    generated_train, train_face[1:5001, :, :, :], train_sp[1:5001, :, :, :], train_lbls[1:5001] = shuffle(generated_train, train_face[1:5001, :, :, :], train_sp[1:5001, :, :, :], train_lbls[1:5001])
    generated_test, test_face, test_sp, test_lbls = shuffle(generated_test, test_face, test_sp, test_lbls)

    # file_name = "gen_img/test/ravdess7_test_real_images_"+source_modality+".png"
    # store_image_maps(test_face[0:90,:,:,:], file_name)
    # file_name = "gen_img/test/ravdess7_test_real_images_"+target_modality+".png"
    # store_image_maps(train_sp[0:90,:,:,:], file_name)
    file_name = "gen_img/test/ravdess_noise_face_lbls_test_gen_images_"+target_modality+".png"
    store_image_maps(generated_train[0:90,:,:,:], file_name)
    pdb.set_trace()

def get_data(datadir):

    # face_classifier = lenet_classifier_model_face(6)
    # face_classifier.trainable = True
    # c_optim = Adagrad(lr=0.00001)
    # face_classifier.compile(loss="categorical_crossentropy", optimizer=c_optim, metrics=['accuracy'])
    # face_classifier.load_weights('temp_files/classifier_28x28_face_new', True)

    dictionary_train = load_obj(datadir + "/models/_full_spec_new_train_112_28.pkl")

    source_train = np.transpose(dictionary_train[source_modality+"_data"] , (0, 3, 1, 2))

    target_train = np.transpose(dictionary_train[target_modality+"_data"], (0, 3, 1, 2))
    lbls_train = dictionary_train[source_modality+"_lbls"]

    dictionary_test = load_obj(datadir + "/models/_full_spec_new_test_112_28.pkl")
    #dictionary_test = load_obj("../models/_full_spec_test_db_v4.pkl")
    source_test = np.transpose(dictionary_test[source_modality+"_data"], (0, 3, 1, 2))
    target_test = np.transpose(dictionary_test[target_modality+"_data"], (0, 3, 1, 2))
    lbls_test = dictionary_test[source_modality+"_lbls"]

    lbls_test = np.reshape(lbls_test, [lbls_test.shape[0], 1])
    lbls_train = np.reshape(lbls_train, [lbls_train.shape[0], 1])

    source_train = (source_train.astype(np.float32) - 127.5)/127.5
    target_train = (target_train.astype(np.float32) - 127.5)/127.5

    source_test = (source_test.astype(np.float32) - 127.5)/127.5
    target_test = (target_test.astype(np.float32) - 127.5)/127.5

    randomize = np.arange(len(source_train))
    np.random.shuffle(randomize)
    source_train = source_train[randomize]
    target_train = target_train[randomize]
    lbls_train = lbls_train[randomize]

    lbls_train = to_categorical(lbls_train)
    lbls_train = lbls_train[:, 1:]
    
    lbls_train[lbls_train == 0] = 0.01
    lbls_train[lbls_train == 1] = 0.99

    lbls_test = to_categorical(lbls_test)
    lbls_test = lbls_test[:, 1:]

    lbls_test[lbls_test == 0] = 0.01
    lbls_test[lbls_test == 1] = 0.99

    return source_train, target_train, lbls_train, source_test, target_test, lbls_test

def crossValidFiles(filename):
    
    #data = load_obj("models/complete_set2.pkl")
    data = load_obj(filename) 
    #{"face_train":pred_train,"train_lbls": Y_train, "specs_train": audio}
    face= data["face_train"]
    audio = data["specs_train"]
    lbls = data["train_lbls"]

    #pdb.set_trace()
    #audio = (audio.astype(np.float32) - 127.5) / 127.5   

    randomize = np.arange(len(face))
    np.random.shuffle(randomize)
    face = face[randomize]
    audio = audio[randomize]
    lbls = lbls[randomize]

    #lbls = lbls.reshape(lbls.shape[0],1)
    #lbls = to_categorical(lbls)
    #lbls = lbls[:, 1:]

    return face, audio, lbls

def lstm_data(target, fold_indx, size_of_feats):

    data = load_obj("../../GANs_models/tmp_dtst/lstm_"+str(size_of_feats)+"_"+str(fold_indx)+".pkl")

    train_feats1= data["feats_train"]
    train_target1 = data["target_train"]
    lbls_train1 = data["lbls_train"]

    data = load_obj("../../GANs_models/tmp_dtst/lstm_"+str(size_of_feats)+"_"+str(fold_indx+1)+".pkl")

    train_feats2= data["feats_train"]
    train_target2 = data["target_train"]
    lbls_train2 = data["lbls_train"]

    data = load_obj("../../GANs_models/tmp_dtst/lstm_"+str(size_of_feats)+"_"+str(fold_indx+2)+".pkl")

    train_feats3= data["feats_train"]
    train_target3 = data["target_train"]
    lbls_train3 = data["lbls_train"]

    train_feats = np.concatenate((train_feats1, train_feats2, train_feats3), axis = 0)
    train_target = np.concatenate((train_target1, train_target2, train_target3), axis =0)
    lbls_train = np.concatenate((lbls_train1, lbls_train2, lbls_train3), axis = 0)

    data = load_obj("../../GANs_models/tmp_dtst/lstm_"+str(size_of_feats)+"_"+str(fold_indx+3)+".pkl")

    test_feats= data["feats_train"]
    test_target = data["target_train"]
    lbls_test = data["lbls_train"]

    data = load_obj("../../GANs_models/tmp_dtst/lstm_"+str(size_of_feats)+"_"+str(fold_indx+4)+".pkl")

    valid_feats= data["feats_train"]
    valid_target = data["target_train"]
    lbls_valid = data["lbls_train"]

    return train_feats, valid_feats, test_feats, train_target, valid_target, test_target, lbls_train, lbls_valid, lbls_test

def load_3d_dataset(target):

    if target == "audio":

        data = load_obj("../../GANs_cnn/models/complete_3d.pkl") 
        #data = load_obj("../../GANs_cnn/models/complete.pkl") 

        train_feats= data["face_train"][:350000]
        train_target = data["specs_train"][:350000]
        lbls_train = data["train_lbls"][:350000]
        train_target = (train_target.astype(np.float32) - 127.5) / 127.5

        valid_feats = data["face_train"][390000:450000]
        valid_target = data["specs_train"][390000:450000]
        lbls_valid = data["train_lbls"][390000:450000]
        valid_target = (valid_target.astype(np.float32) - 127.5) / 127.5

        test_feats = data["face_train"][500000:]
        test_target = data["specs_train"][500000:]
        lbls_test= data["train_lbls"][500000:]
        test_target = (test_target.astype(np.float32) - 127.5) / 127.5
        
        randomize = np.arange(len(train_feats))
        np.random.shuffle(randomize)
        train_feats = train_feats[randomize]
        train_target = train_target[randomize]
        lbls_train = lbls_train[randomize]

        reshape_train = lbls_train[:,0].reshape(lbls_train.shape[0],1)
        lbls_train = np.concatenate((lbls_train, lbls_train, reshape_train), axis=1)

        lbls_train[lbls_train == 0] = 0.01
        lbls_train[lbls_train == 1] = 0.99

        reshape_test = lbls_test[:,0].reshape(lbls_test.shape[0],1)
        audio_lbls_test = np.concatenate((lbls_test, lbls_test, reshape_test), axis=1)

        lbls_test[lbls_test == 0] = 0.01
        lbls_test[lbls_test == 1] = 0.99

        reshape_valid = lbls_valid[:,0].reshape(lbls_valid.shape[0],1)
        lbls_valid = np.concatenate((lbls_valid, lbls_valid, reshape_valid), axis=1)

        lbls_test[lbls_test == 0] = 0.01
        lbls_test[lbls_test == 1] = 0.99

        return train_feats, train_target, lbls_train, valid_feats, valid_target, lbls_valid, test_feats, test_target, lbls_test
    
    elif target == "face":

        data = load_obj("models/extrAudioFeats64.pkl") 

        X_train= data["trainFeats"]
        Y_train = data["face_train"]
        audio_lbls_train = data["train_lbls"]
    
        X_test= data["testFeats"]
        Y_test = data["face_test"]
        
        audio_lbls_test= data["test_lbls"]
        
        randomize = np.arange(len(X_train))
        np.random.shuffle(randomize)
        X_train = X_train[randomize]
        Y_train = Y_train[randomize]
        audio_lbls_train = audio_lbls_train[randomize]

        reshape_train = audio_lbls_train[:,0].reshape(audio_lbls_train.shape[0],1)
        audio_lbls_train = np.concatenate((audio_lbls_train, audio_lbls_train, reshape_train), axis=1)

        audio_lbls_train[audio_lbls_train == 0] = 0.01
        audio_lbls_train[audio_lbls_train == 1] = 0.99

        reshape_test = audio_lbls_test[:,0].reshape(audio_lbls_test.shape[0],1)
        audio_lbls_test = np.concatenate((audio_lbls_test, audio_lbls_test, reshape_test), axis=1)

        audio_lbls_test[audio_lbls_test == 0] = 0.01
        audio_lbls_test[audio_lbls_test == 1] = 0.99

        Y_train = Y_train.transpose(0,2,3,1)
        Y_test = Y_test.transpose(0,2,3,1)
        
        return X_train, Y_train, audio_lbls_train, X_test, Y_test, audio_lbls_test


            
        
def generated_data_dacsGANs(X_train, X_test, LABEL_train,  LABEL_test, Y_test, generator):

    noise_train = np.random.normal(0, 1, (X_train.shape[0], 100))
    noise_train = np.concatenate([noise_train, LABEL_train[:,0:6]], axis = 1)

    noise_test = np.random.normal(0, 1, (X_test.shape[0], 100))
    noise_test = np.concatenate([noise_test, LABEL_test[:,0:6]], axis = 1)

    generated_test = generator.predict([noise_test])
    generated_images1 = generator.predict([noise_train])
    
    gen_data_dictionary = {"gen_train": generated_images1, 
                           "gen_train_lbls": LABEL_train, 
                           "gen_test": generated_test, 
                           "gen_test_lbls": LABEL_test}
    store_obj("temp_files/gen_data_.pkl", gen_data_dictionary)


def _generated_data_wgans_(self, face_train, audio_train, face_test, audio_test, lbls_train, lbls_test):

        generator = self.generator
        generator.load_weights('temp_files/iwgans_gen_audio_only_noise__')
        noise = np.random.normal(0, 1, (face_test.shape[0], 64))
        #pdb.set_trace()
        #generated_test = generator.predict([np.concatenate([ lbls_test, noise], axis = 1)])
        generated_test = generator.predict(lbls_test)
        noise = np.random.normal(0, 1, (face_train.shape[0], 64))
        #generated_train = generator.predict([np.concatenate([lbls_train, noise], axis = 1)])
        generated_train = generator.predict(lbls_train)
        generated_train = shuffle(generated_train)


        #generated_train = generator.predict(face_train)
        #generated_test = generator.predict(face_test)

        file_name = "gen_img/test/test_specs_tmp_lbls_noise.png"
        #self.store_image_maps(generated_train[0:100,:,:,:], file_name)

        gen_data_dictionary = {"gen_train": generated_train,
                               "real_train_face": face_train,
                               "real_train_audio": audio_train,
                               "gen_train_lbls": lbls_train,
                               "real_test_audio": audio_test,
                               "real_test_face": face_test,
                               "gen_test": generated_test, 
                               "gen_test_lbls": lbls_test}

        store_obj("temp_files/gen_data.pkl", gen_data_dictionary)

def load_3d_dataset_wgans():

        data = load_obj("../models/complete.pkl")
        face_train= data["face_train"]
        spec_train = data["specs_train"]

        lbls_train = data["train_lbls"]
        spec_train = (spec_train.astype(np.float32) - 127.5) / 127.5

        face_test= data["face_test"]
        spec_test = data["specs_test"]
        lbls_test= data["test_lbls"]
        spec_test = (spec_test.astype(np.float32) - 127.5) / 127.5

        return face_train, spec_train, lbls_train, face_test, spec_test, lbls_test


def load_real_samples_RAV():

#     face_classifier = lenet_classifier_model_face(6)
#     face_classifier.trainable = True
#     c_optim = Adagrad(lr=0.00001)
#     face_classifier.compile(loss="categorical_crossentropy", optimizer=c_optim, metrics=['accuracy'])
#     face_classifier.load_weights('_classifier_RAV_face_', True)
    
    cnn_dict= load_obj("../models/rav1.pkl")

    train_face = cnn_dict["faces_Tr"]
    train_sp = cnn_dict["specs_Tr"]
    train_lbls = cnn_dict["ann_Tr"]
    
    test_face = cnn_dict["faces_C"]
    test_sp = cnn_dict["specs_C"]
    test_lbls = cnn_dict["ann_C"]

    conf_face = cnn_dict["faces_Ts"]
    conf_sp = cnn_dict["specs_Ts"]
    conf_lbls = cnn_dict["ann_Ts"]
    
    train_sp = train_sp.transpose(0,3,1,2)
    test_sp = test_sp.transpose(0,3,1,2)
    conf_sp = conf_sp.transpose(0,3,1,2)

    train_face = train_face.transpose(0,3,1,2)
    test_face = test_face.transpose(0,3,1,2)
    conf_face = conf_face.transpose(0,3,1,2)

    #train_lbls = face_classifier.predict_classes(train_face) + 1
    #test_lbls = face_classifier.predict_classes(test_face) + 1
    #conf_lbls = face_classifier.predict_classes(conf_face) + 1

    train_lbls = train_lbls.astype(int)
    test_lbls = test_lbls.astype(int)
    conf_lbls = conf_lbls.astype(int)

    train_lbls = to_categorical(train_lbls)
    train_lbls = train_lbls[:, 1:]
    train_lbls = np.concatenate((train_lbls, train_lbls, train_lbls[:,0].reshape(train_lbls.shape[0],1)),axis=1)

    test_lbls = to_categorical(test_lbls)
    test_lbls = test_lbls[:, 1:]
    test_lbls = np.concatenate((test_lbls,test_lbls, test_lbls[:,0].reshape(test_lbls.shape[0],1)), axis=1)

    conf_lbls = to_categorical(conf_lbls)
    conf_lbls = conf_lbls[:, 1:]
    conf_lbls = np.concatenate((conf_lbls,conf_lbls, conf_lbls[:,0].reshape(conf_lbls.shape[0],1)), axis=1)

    train_face = (train_face.astype(np.float32) - 127.5)/127.5
    train_sp = (train_sp.astype(np.float32) - 127.5)/127.5

    test_face = (test_face.astype(np.float32) - 127.5)/127.5
    test_sp = (test_sp.astype(np.float32) - 127.5)/127.5

    conf_face = (conf_face.astype(np.float32) - 127.5)/127.5
    conf_sp = (conf_sp.astype(np.float32) - 127.5)/127.5
    
    return train_face, train_sp, train_lbls, test_face, test_sp, test_lbls, conf_face, conf_sp, conf_lbls


def load_3d_iwGANs():

    data = load_obj("../models/complete.pkl") 

    face_train= data["face_train"]
    spec_train = data["specs_train"]
    lbls_train = data["train_lbls"]
    spec_train = (spec_train.astype(np.float32) - 127.5) / 127.5
    
    face_test= data["face_test"]
    spec_test = data["specs_test"]
    lbls_test= data["test_lbls"]
    spec_test = (spec_test.astype(np.float32) - 127.5) / 127.5
    
    return face_train, spec_train, lbls_train, face_test, spec_test, lbls_test


def load_faces_from_dict():
    
    from sklearn.utils import shuffle

    face_dict = load_obj("models/gen_with_wgans_face.pkl",)
    gen_train = face_dict["gen_train"]
    real_train = face_dict["real_train_face"]
    #pdb.set_trace()
    gen_train = shuffle(gen_train)
    real_train = shuffle(real_train)

    for i in range(0,2998):
        generated_img = gen_train[i,:,:,:]
        generated_img = generated_img * 127.5 + 127.5
        generated_img = cv2.resize(generated_img, dsize = (224, 224), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('generated_imgs/face/generated/generated_img_'+str(i)+'.jpg', generated_img)

        real_img = real_train[i,:,:,:]
        real_img = real_img * 127.5 + 127.5
        real_img = cv2.resize(real_img, dsize = (224, 224), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('generated_imgs/face/real/real_img_'+str(i)+'.jpg', real_img)

def gen_data_iwGANs(generator, face_train, audio_train, face_test, audio_test, lbls_train, lbls_test):

        from sklearn.utils import shuffle
        generator.load_weights('models/iwgans_gen_specs_')

        (face_test, audio_test, lbls_test) = shuffle(face_test, audio_test, lbls_test)
        face_test = face_test[0:30000]
        audio_test = audio_test[0:30000]
        lbls_test = lbls_test[0:30000]

        noise = np.random.normal(0, 1, (face_test.shape[0], 100))
        noise = np.concatenate([noise, lbls_test], axis = 1) #noise = face_imgs

        generated_test = generator.predict([noise])
        (face_train, audio_train, lbls_train) = shuffle(face_train, audio_train, lbls_train)
        face_train = face_train[0:300000]
        audio_train = audio_train[0:300000]
        lbls_train = lbls_train[0:300000]

        #pdb.set_trace()
        noise = np.random.normal(0, 1, (face_train.shape[0], 100))
        noise = np.concatenate([noise, lbls_train], axis = 1) #noise = face_imgs
        generated_train = generator.predict([noise])
        generated_train = shuffle(generated_train)

        pdb.set_trace()
        gen_data_dictionary = {"gen_train": generated_train, "gen_train_lbls": lbls_train, "gen_test": generated_test, "gen_test_lbls": lbls_test}
        store_obj("models/gen_with_wgans_sp.pkl", gen_data_dictionary)
        real_data_dictionary = {"real_train_audio": audio_train, "real_test_audio": audio_test, "train_lbls": lbls_train, "test_lbls": lbls_test}
        store_obj("models/real.pkl", real_data_dictionary)

def create_db(folder_path):

        all_files = os.listdir(folder_path)
        
        set1 = []
        set2 = []
        set3 = []
        set4 = []
        set5 = []
        set6 = []
        set7 = []
        
        for face in all_files:
           split_face_name = face.split("_")
           if int(split_face_name[2]) <= 1060: 
                set1.append(face)
           elif int(split_face_name[2]) > 1061 and int(split_face_name[2]) <= 1075:
                set2.append(face)
           elif int(split_face_name[2]) > 1076 and int(split_face_name[2]) <= 1095:
                set3.append(face)

        set_list = {"train_set":set1, "valid_set":set2, "test_set":set3}
        for set_ in set_list:

            face_set = []
            audio_set = []
            lbls_set= []       
            
            for file in set_list[set_]:   
              obj = load_obj(folder_path+ file)
              face_data = obj["face_data"]
              for face in face_data:
                face_set.append(face)
              audio_data = obj["audio_data"]
              for audio in audio_data:
                audio_set.append(audio)
              lbls = obj["lbls"]
              for lbl in lbls:
                lbls_set.append(lbl)

            pdb.set_trace()    
            dictionary = {"face": np.asarray(face_set), "audio": np.asarray(audio_set), "lbls": np.asarray(lbls_set)  }
            store_obj("models/dataset_"+ set_+"_.pkl", dictionary)
            #pdb.set_trace()


def aggregate_files(input_file):

    # Check if the input file is a string or list-like
    if isinstance(input_file, str):
        train_array = crossValidFiles(input_file)
    else:
        pdb.set_trace()
        train_array = np.concatenate([crossValidFiles(f) for f in input_file], axis=0)

    # rest of your code to create variables 'data' and 'labels'
    pdb.set_trace()
    return data, labels


def permutations():

        sets = ['models/temp_models/complete_1.pkl', 'models/temp_models/complete_2.pkl', 'models/temp_models/complete_3.pkl', 'models/temp_models/complete_4.pkl',
            'models/temp_models/complete_5.pkl', 'models/temp_models/complete_6.pkl', 'models/temp_models/complete_7.pkl']
        
        for i in range(0,7):
                
             train_set = sets[:i]+sets[i+1:]    
             train_sets = [crossValidFiles(item) for item in train_set]
             train_face = np.concatenate([a for (a, b, c) in train_sets], axis=0)[0:300000]
             train_audio = np.concatenate([b for (a, b, c) in train_sets], axis=0)[0:300000]
             train_lbls = np.concatenate([c for (a, b, c) in train_sets], axis=0)[0:300000]
             test_face, test_audio, test_lbls = crossValidFiles(sets[i])
                

#create_db("../../databases/CREMA_D/data/seq_/")      
#$permutations()                        
                               
                               

        
