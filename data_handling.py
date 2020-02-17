import pickle
import numpy as np
from keras.utils import to_categorical
import pdb
from keras_models import *
from keras.optimizers import SGD, Adagrad
import cv2
import os


def _norm_(image):

    return (image.astype(np.float32) - 127.5)/127.5

def _trnse_(_arr_):

    return np.transpose(_arr_ , (0, 3, 1, 2))


class data_handle():

    def __init__(self):

        self.src_mod = "face"
        self.trg_mod = "audio"

        self.dacss_train_path = "../../GANs_cnn/models/_full_spec_new_train_112_28.pkl"
        self.dacss_test_path = "../../GANs_cnn/models/_full_spec_new_test_112_28.pkl"
        self.db_3d = "../../GANs_cnn/models/complete.pkl"
        self.wgans_path = "../models/complete.pkl"

        self._indx_ = 180000
        self._indx2_= 215000

        self._lmt_1 = 350000
        self._lmt_2 = 390000
        self._lmt_3 = 450000
        self._lmt_4 = 500000

    
    def _sft_crisp_lbl_(self, _my_arr_):

        _my_arr_[_my_arr_ == 0] = 0.01
        _my_arr_[_my_arr_ == 1] = 0.99

        return _my_arr_
    
    def load_obj(self, filename):
        with open(filename, "rb") as fp:  # Unpickling
            return pickle.load(fp, encoding='bytes') 


    def store_obj(self, filename, obj):
        with open(filename, "wb") as fp:  # Pickling
            pickle.dump(obj, fp, protocol=4)


    def get_data(self, datadir):

        _dct_trn_ = self.load_obj(self.dacss_train_path)

        _src_trn_ = np.transpose(_dct_trn_[self.src_mod+"_data"] , (0, 3, 1, 2))

        _trg_trn_ = np.transpose(_dct_trn_[self.trg_mod+"_data"], (0, 3, 1, 2))
        _lbls_trn_ = _dct_trn_[self.src_mod+"_lbls"]

        _dct_tst_ = self.load_obj(self.dacss_test_path)
        _src_tst_ = np.transpose(_dct_tst_[self.src_mod+"_data"], (0, 3, 1, 2))
        _trg_tst_ = np.transpose(_dct_tst_[self.trg_mod+"_data"], (0, 3, 1, 2))
        _lbls_tst_ = _dct_tst_[self.src_mod+"_lbls"]

        _lbls_tst_ = np.reshape(_lbls_tst_, [_lbls_tst_.shape[0], 1])
        _lbls_trn_ = np.reshape(_lbls_trn_, [_lbls_trn_.shape[0], 1])

        _src_trn_ = _norm_(_src_trn_)    
        _trg_trn_ = _norm_(_trg_trn_)

        _src_tst_ = _norm_(_src_tst_)
        _trg_tst_ = _norm_(_trg_tst_)

        _src_vld_ = _src_trn_[self._indx2_:]
        _trg_vld_ = _trg_trn_[self._indx2_:]
        _lbls_vld_ = _lbls_trn_[self._indx2_:]

        _src_trn_ = _src_trn_[:self._indx_]
        _trg_trn_ = _trg_trn_[:self._indx_]
        _lbls_trn_ = _lbls_trn_[:self._indx_]

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
        
        _lbls_trn_ = self._sft_crisp_lbl_(_lbls_trn_)

        _lbls_vld_ = to_categorical(_lbls_vld_)
        _lbls_vld_ = _lbls_vld_[:, 1:]
        
        _lbls_vld_ = self._sft_crisp_lbl_(_lbls_vld_)

        _lbls_tst_ = to_categorical(_lbls_tst_)
        _lbls_tst_ = _lbls_tst_[:, 1:]

        _lbls_tst_ = self._sft_crisp_lbl_(_lbls_tst_)
        
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



    def temporal_feats(
            self,
            target, 
            fold_indx, 
            size_of_feats, 
            feats_type, 
            db_path):

        _dt_ = self.load_obj(db_path+
            feats_type+"_"+
            str(size_of_feats)+"_"+
            str(fold_indx)+
            ".pkl")

        train_feats= _dt_["feats_train"]
        train_target = _dt_["target_train"]
        lbls_train = _dt_["lbls_train"]


        _dt_ = self.load_obj(db_path+
            feats_type+"_"+
            str(size_of_feats)+"_"+
            str(2)+
            ".pkl")

        test_feats= _dt_["feats_train"]
        test_target = _dt_["target_train"]
        lbls_test = _dt_["lbls_train"]

        _dt_ = self.load_obj(db_path+
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


    def load_3d_dataset(self, target):

        if target == "audio":
            
            _dt_ = self.load_obj(self.db_3d)

           
            trn_fts= _dt_["face_train"][:self._lmt_1]
            trn_trg = _dt_["specs_train"][:self._lmt_1]
            trn_lbls = _dt_["train_lbls"][:self._lmt_1]
            trn_trg = _norm_(trn_trg)

            
            vld_fts = _dt_["face_train"][self._lmt_2:self._lmt_3]
            vld_trg = _dt_["specs_train"][self._lmt_2:self._lmt_3]
            vld_lbls = _dt_["train_lbls"][self._lmt_2:self._lmt_3]
            
            vld_trg = _norm_(vld_trg)

            tst_fts = _dt_["face_train"][self._lmt_4:]
            tst_trg = _dt_["specs_train"][self._lmt_4:]
            tst_lbls= _dt_["train_lbls"][self._lmt_4:]
            tst_trg = _norm_(tst_trg)
            
            randomize = np.arange(len(trn_fts))
            np.random.shuffle(randomize)
            trn_fts = trn_fts[randomize]
            trn_trg = trn_trg[randomize]
            trn_lbls = trn_lbls[randomize]

            trn_lbls = self._sft_crisp_lbl_(trn_lbls)
            tst_lbls = self._sft_crisp_lbl_(tst_lbls)
            vld_lbls = self._sft_crisp_lbl_(vld_lbls)
            
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

            data = self.load_obj("models/extrAudioFeats64.pkl") 

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

            trn_lbls = self._sft_crisp_lbl_(trn_lbls)
            tst_lbls = self._sft_crisp_lbl_(tst_lbls)

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

                               

        
