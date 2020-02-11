import pickle
import numpy as np
from keras.utils import to_categorical
import pdb
from keras_models import *
from keras.optimizers import SGD, Adagrad
import cv2
import os

src_mod = "face"
trg_mod = "audio"

dacss_train_path = "../../GANs_cnn/models/_full_spec_new_train_112_28.pkl"
dacss_test_path = "../../GANs_cnn/models/_full_spec_new_test_112_28.pkl"
db_3d = "../../GANs_cnn/models/complete_3d.pkl"

_indx_ = 200000
_indx2_ = 215000


def load_obj(filename):
    with open(filename, "rb") as fp:  # Unpickling
        return pickle.load(fp, encoding='bytes') 


def store_obj(filename, obj):
    with open(filename, "wb") as fp:  # Pickling
        pickle.dump(obj, fp, protocol=4)

def _norm_(image):

    return (image.astype(np.float32) - 127.5)/127.5


def get_data(datadir):

    _dct_trn_ = load_obj(dacss_train_path)

    _src_trn_ = np.transpose(_dct_trn_[src_mod+"_data"] , (0, 3, 1, 2))

    _trg_trn_ = np.transpose(_dct_trn_[trg_mod+"_data"], (0, 3, 1, 2))
    _lbls_trn_ = _dct_trn_[src_mod+"_lbls"]

    _dct_tst_ = load_obj(dacss_test_path)
    _src_tst_ = np.transpose(_dct_tst_[src_mod+"_data"], (0, 3, 1, 2))
    _trg_tst_ = np.transpose(_dct_tst_[trg_mod+"_data"], (0, 3, 1, 2))
    _lbls_tst_ = _dct_tst_[src_mod+"_lbls"]

    _lbls_tst_ = np.reshape(_lbls_tst_, [_lbls_tst_.shape[0], 1])
    _lbls_trn_ = np.reshape(_lbls_trn_, [_lbls_trn_.shape[0], 1])

    _src_trn_ = _norm_(_src_trn_)    
    _trg_trn_ = _norm_(_trg_trn_)

    _src_tst_ = _norm_(_src_tst_)
    _trg_tst_ = _norm_(_trg_tst_)

    _src_vld_ = _src_trn_[_indx2_:]
    _trg_vld_ = _trg_trn_[_indx2_:]
    _lbls_vld_ = _lbls_trn_[_indx2_:]

    _src_trn_ = _src_trn_[:_indx_]
    _trg_trn_ = _trg_trn_[:_indx_]
    _lbls_trn_ = _lbls_trn_[:_indx_]

    _rndz_ = np.arange(len(_src_trn_))
    np.random.shuffle(_rndz_)
    
    _src_trn_ = _src_trn_[_rndz_]
    _trg_trn_ = _trg_trn_[_rndz_]
    _lbls_trn_ = _lbls_trn_[_rndz_]

    _rndz_ = np.arange(len(_src_vld_))
    np.random.shuffle(_rndz_)

    _src_vld_ = _src_vld_[_rndz_]
    _trg_vld_ = _trg_vld_[_rndz_]
    _lbls_vld_ = _lbls_vld_[_rndz_]

    _lbls_trn_ = to_categorical(_lbls_trn_)
    _lbls_trn_ = _lbls_trn_[:, 1:]
    
    _lbls_trn_[_lbls_trn_ == 0] = 0.01
    _lbls_trn_[_lbls_trn_ == 1] = 0.99

    _lbls_vld_ = to_categorical(_lbls_vld_)
    _lbls_vld_ = _lbls_vld_[:, 1:]
    
    _lbls_vld_[_lbls_vld_ == 0] = 0.01
    _lbls_vld_[_lbls_vld_ == 1] = 0.99

    _lbls_tst_ = to_categorical(_lbls_tst_)
    _lbls_tst_ = _lbls_tst_[:, 1:]

    _lbls_tst_[_lbls_tst_ == 0] = 0.01
    _lbls_tst_[_lbls_tst_ == 1] = 0.99

    del _dct_trn_, _dct_tst_

    _dct_ = {"_src_trn_": _src_trn_,
        "_trg_trn_": _trg_trn_,
        "_lbls_trn_": _lbls_trn_,
        "_src_vld_": _src_vld_,
        "_trg_vld_": _trg_vld_,
        "_lbls_vld_": _lbls_vld_,
        "_src_tst_": _src_tst_,
        "_trg_tst_": _trg_tst_,
        "_lbls_tst_": _lbls_tst_}

    return _dct_

def crossValidFiles(filename):
    
    #data = load_obj("models/complete_set2.pkl")
    data = load_obj(filename) 
    #{"face_train":pred_train,"train_lbls": Y_train, "specs_train": audio}
    face= data["face_train"]
    audio = data["specs_train"]
    lbls = data["train_lbls"]

    randomize = np.arange(len(face))
    np.random.shuffle(randomize)
    face = face[randomize]
    audio = audio[randomize]
    lbls = lbls[randomize]

    return face, audio, lbls


def temporal_feats(target, 
        fold_indx, 
        size_of_feats, 
        feats_type, 
        db_path):

    _dt_ = load_obj(db_path+
        feats_type+"_"+
        str(size_of_feats)+"_"+
        str(fold_indx)+
        ".pkl")

    train_feats= _dt_["feats_train"]
    train_target = _dt_["target_train"]
    lbls_train = _dt_["lbls_train"]


    _dt_ = load_obj(db_path+
        feats_type+"_"+
        str(size_of_feats)+"_"+
        str(2)+
        ".pkl")

    test_feats= _dt_["feats_train"]
    test_target = _dt_["target_train"]
    lbls_test = _dt_["lbls_train"]

    _dt_ = load_obj(db_path+
        feats_type+"_"+
        str(size_of_feats)+"_"+
        str(2)+
        ".pkl")

    valid_feats= _dt_["feats_train"]
    valid_target = _dt_["target_train"]
    lbls_valid = _dt_["lbls_train"]

    _dct_ = {"train_feats": train_feats, 
             "valid_feats": valid_feats, 
             "test_feats": test_feats,
             "train_target": train_target, 
             "valid_target": valid_target, 
             "test_target": test_target,
             "lbls_train": lbls_train, 
             "lbls_valid": lbls_valid, 
             "lbls_test": lbls_test,
             }

    return _dct_


def load_3d_dataset(target):

    if target == "audio":
        data = load_obj(db_3d) 

        trn_fts= data["face_train"][:350000]
        trn_trg = data["specs_train"][:350000]
        trn_lbls = data["train_lbls"][:350000]
        trn_trg = _norm_(trn_trg)

        vld_fts = data["face_train"][390000:450000]
        vld_trg = data["specs_train"][390000:450000]
        vld_lbls = data["train_lbls"][390000:450000]
        
        vld_trg = _norm_(vld_trg)

        tst_fts = data["face_train"][500000:]
        tst_trg = data["specs_train"][500000:]
        tst_lbls= data["train_lbls"][500000:]
        tst_trg = _norm_(tst_trg)
        
        randomize = np.arange(len(trn_fts))
        np.random.shuffle(randomize)
        trn_fts = trn_fts[randomize]
        trn_trg = trn_trg[randomize]
        trn_lbls = trn_lbls[randomize]

        trn_lbls[trn_lbls == 0] = 0.01
        trn_lbls[trn_lbls == 1] = 0.99

        tst_lbls[tst_lbls == 0] = 0.01
        tst_lbls[tst_lbls == 1] = 0.99

        vld_lbls[tst_lbls == 0] = 0.01
        vld_lbls[tst_lbls == 1] = 0.99
       
        _dct_ = {"trn_fts": trn_fts, 
             "trn_trg": trn_trg, 
             "trn_lbls": trn_lbls,
             "vld_fts": vld_fts, 
             "vld_trg": vld_trg, 
             "vld_lbls": vld_lbls,
             "tst_fts": tst_fts, 
             "tst_trg": tst_trg, 
             "tst_lbls": tst_lbls,
             }

        return _dct_

    
    elif target == "face":

        data = load_obj("models/extrAudioFeats64.pkl") 

        trn_feats= data["trainFeats"]
        trn_tgt = data["face_train"]
        trn_lbls = data["train_lbls"]
    
        tst_fts= data["testFeats"]
        tst_trg = data["face_test"]
        
        tst_lbls= data["test_lbls"]
        
        randomize = np.arange(len(trn_feats))
        np.random.shuffle(randomize)
        trn_feats = trn_feats[randomize]
        trn_tgt = trn_tgt[randomize]
        trn_lbls = trn_lbls[randomize]

        trn_lbls[trn_lbls == 0] = 0.01
        trn_lbls[trn_lbls == 1] = 0.99

        tst_lbls[tst_lbls == 0] = 0.01
        tst_lbls[tst_lbls == 1] = 0.99

        trn_tgt = trn_tgt.transpose(0,2,3,1)
        tst_trg = tst_trg.transpose(0,2,3,1)

        _dct_ = {"trn_feats": trn_feats, 
             "trn_tgt": trn_tgt, 
             "trn_lbls": trn_lbls,
             "tst_fts": tst_fts, 
             "tst_trg": tst_trg, 
             "tst_lbls": tst_lbls,
             }
        
        return _dct_


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
    
    return face_train,\
        spec_train,\
        lbls_train,\
        face_test,\
        spec_test,\
        lbls_test


def gen_data_iwGANs(generator, 
        face_train, 
        audio_train, 
        face_test, 
        audio_test, 
        lbls_train, 
        lbls_test):

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


                               

        
