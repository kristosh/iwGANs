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

        self.cr_train_path = "../../GANs_cnn/models/_full_spec_new_train_112_28.pkl"
        self.cr_test_path = "../../GANs_cnn/models/_full_spec_new_test_112_28.pkl"

        self.rv_train_path = "../../GANs_cnn/models/ravdess/trnNseq/train_non_seq_rav_0.pkl"
        self.rv_test_path = "../../GANs_cnn/models/ravdess/test_non_seq_rav.pkl"

        self.db_3d = "../../GANs_cnn/models/complete.pkl"
        self.db_3d_ravdess = "../../GANs_cnn/models/ravdess/3d/"
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


    def get_data_cp(self, datadir, path,  db):

        _dct_ = self.load_obj(path + 
            "_full_spec_new_train_112_28.pkl")

        _src_trn_ = np.transpose(
            _dct_[self.src_mod+"_data"], (0, 3, 1, 2))
        _trg_trn_ = np.transpose(
            _dct_[self.trg_mod+"_data"], (0, 3, 1, 2))
        _lbls_trn_ = _dct_[self.src_mod+"_lbls"]

        _dct_ = self.load_obj(path + 
            "_full_spec_new_test_112_28.pkl")

        _src_tst_ = np.transpose(
            _dct_[self.src_mod+"_data"], (0, 3, 1, 2))
        _trg_tst_ = np.transpose(
            _dct_[self.trg_mod+"_data"], (0, 3, 1, 2))
        _lbls_tst_ = _dct_[self.src_mod+"_lbls"]

        _lbls_tst_ = np.reshape(
            _lbls_tst_, 
            [_lbls_tst_.shape[0], 1])
        _lbls_trn_ = np.reshape(
            _lbls_trn_, 
            [_lbls_trn_.shape[0], 1])

        _src_trn_ = _norm_(_src_trn_)
        _trg_trn_ = _norm_(_trg_trn_)

        _src_tst_ = _norm_(_src_tst_)
        _trg_tst_ = _norm_(_trg_tst_)

        rndz = np.arange(len(_src_trn_))
        np.random.shuffle(rndz)

        _src_trn_ = _src_trn_[rndz]
        _trg_trn_ = _trg_trn_[rndz]
        _lbls_trn_ = _lbls_trn_[rndz]

        _lbls_trn_ = to_categorical(_lbls_trn_)
        _lbls_trn_ = _lbls_trn_[:, 1:]
        
        _lbls_trn_ = self._sft_crisp_lbl_(_lbls_trn_)

        _lbls_tst_ = to_categorical(_lbls_tst_)
        _lbls_tst_ = _lbls_tst_[:, 1:]

        _lbls_tst_ = self._sft_crisp_lbl_(_lbls_tst_)

        _dct_ = {"_src_trn_": _src_trn_,
                "_trg_trn_": _trg_trn_,
                "_lbls_trn_": _lbls_trn_,
                "_src_tst_": _src_tst_,
                "_trg_tst_": _trg_tst_,
                "_lbls_tst_": _lbls_tst_}

        return _dct_

    
    def get_data_rv(self, datadir, path,  db):
        
        _dct_ = self.load_obj(path+db)

        _src_trn_ = np.transpose(_dct_[self.src_mod+"_data"] , (0, 3, 1, 2))

        _trg_trn_ = np.transpose(_dct_[self.trg_mod+"_data"], (0, 3, 2, 1))
        _lbls_trn_ = _dct_[self.src_mod+"_lbls"]

        _dct_ = self.load_obj(self.rv_test_path)
        _src_tst_ = np.transpose(_dct_[self.src_mod+"_data"], (0, 3, 1, 2))
        _trg_tst_ = np.transpose(_dct_[self.trg_mod+"_data"], (0, 3, 2, 1))
        _lbls_tst_ = _dct_[self.src_mod+"_lbls"]

        _lbls_tst_ = np.reshape(_lbls_tst_, [_lbls_tst_.shape[0], 1])
        _lbls_trn_ = np.reshape(_lbls_trn_, [_lbls_trn_.shape[0], 1])
        
        _rndz_ = np.arange(len(_src_trn_))
        np.random.shuffle(_rndz_)
        
        _src_trn_ = _src_trn_[_rndz_]
        _trg_trn_ = _trg_trn_[_rndz_]
        _lbls_trn_ = _lbls_trn_[_rndz_]

        _lbls_trn_ = to_categorical(_lbls_trn_)
        _lbls_trn_ = _lbls_trn_[:, 1:]
        
        _lbls_trn_ = self._sft_crisp_lbl_(_lbls_trn_)

        _lbls_tst_ = to_categorical(_lbls_tst_)
        _lbls_tst_ = _lbls_tst_[:, 1:]

        _lbls_tst_ = self._sft_crisp_lbl_(_lbls_tst_)

        _dct_ = {"_src_trn_": _src_trn_,
            "_trg_trn_": _trg_trn_,
            "_lbls_trn_": _lbls_trn_,
            "_src_tst_": _norm_(_src_tst_),
            "_trg_tst_": _norm_(_trg_tst_),
            "_lbls_tst_": _lbls_tst_}

        return _dct_


    def get_data_rv(self, datadir, path,  db):
        
        _dct_ = self.load_obj(path+db)

        pdb.set_trace()
        
        _src_trn_ = np.transpose(_dct_[self.src_mod+"_data"] , (0, 3, 1, 2))

        _trg_trn_ = np.transpose(_dct_[self.trg_mod+"_data"], (0, 3, 2, 1))
        _lbls_trn_ = _dct_[self.src_mod+"_lbls"]

        _dct_ = self.load_obj(self.rv_test_path)
        _src_tst_ = np.transpose(_dct_[self.src_mod+"_data"], (0, 3, 1, 2))
        _trg_tst_ = np.transpose(_dct_[self.trg_mod+"_data"], (0, 3, 2, 1))
        _lbls_tst_ = _dct_[self.src_mod+"_lbls"]

        _lbls_tst_ = np.reshape(_lbls_tst_, [_lbls_tst_.shape[0], 1])
        _lbls_trn_ = np.reshape(_lbls_trn_, [_lbls_trn_.shape[0], 1])
        
        _rndz_ = np.arange(len(_src_trn_))
        np.random.shuffle(_rndz_)
        
        _src_trn_ = _src_trn_[_rndz_]
        _trg_trn_ = _trg_trn_[_rndz_]
        _lbls_trn_ = _lbls_trn_[_rndz_]

        _lbls_trn_ = to_categorical(_lbls_trn_)
        _lbls_trn_ = _lbls_trn_[:, 1:]
        
        _lbls_trn_ = self._sft_crisp_lbl_(_lbls_trn_)

        _lbls_tst_ = to_categorical(_lbls_tst_)
        _lbls_tst_ = _lbls_tst_[:, 1:]

        _lbls_tst_ = self._sft_crisp_lbl_(_lbls_tst_)

        _dct_ = {"_src_trn_": _src_trn_,
            "_trg_trn_": _trg_trn_,
            "_lbls_trn_": _lbls_trn_,
            "_src_tst_": _norm_(_src_tst_),
            "_trg_tst_": _norm_(_trg_tst_),
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
            str(size_of_feats)+"_da_"+
            str(fold_indx)+
            ".pkl")

        train_feats= _dt_["feats_train"]
        train_target = _dt_["target_train"]
        lbls_train = _dt_["lbls_train"]


        _dt_ = self.load_obj(db_path+
            feats_type+"_"+
            str(size_of_feats)+"_da_"+
            str("test")+
            ".pkl")

        test_feats= _dt_["feats_train"]
        test_target = _dt_["target_train"]
        lbls_test = _dt_["lbls_train"]

        _dt_ = self.load_obj(db_path+
            feats_type+"_"+
            str(size_of_feats)+"_da_"+
            str("test")+
            ".pkl")

        valid_feats= _dt_["feats_train"]
        valid_target = _dt_["target_train"]
        lbls_valid = _dt_["lbls_train"]

        _dct_ = {"trn_fts": train_feats, 
                "vld_fts": valid_feats, 
                "tst_fts": test_feats,
                "trn_trg": train_target, 
                "vld_trg": valid_target, 
                "tst_trg": test_target,
                "trn_lbls": lbls_train, 
                "vld_lbls": lbls_valid, 
                "tst_lbls": lbls_test,
                }

        return _dct_


    def load_3d_dataset_rav(self, target):
        
        files = os.listdir(self.db_3d_ravdess)
            
        _dt_train = self.load_obj(self.db_3d_ravdess + files[0])

        trn_fts= _dt_train["feats_train"]
        trn_trg = _dt_train["target_train"]
        trn_lbls = _dt_train["lbls_train"]
        trn_trg = _norm_(trn_trg)

        tst_fts = _dt_train["feats_tst"]
        tst_trg = _dt_train["target_test"]
        tst_lbls= _dt_train["lbls_test"]
        tst_trg = _norm_(tst_trg)
        
        randomize = np.arange(len(trn_fts))
        np.random.shuffle(randomize)
        trn_fts = trn_fts[randomize]
        trn_trg = trn_trg[randomize]
        trn_lbls = trn_lbls[randomize]

        randomize = np.arange(len(tst_fts))
        np.random.shuffle(randomize)
        tst_fts = tst_fts[randomize]
        tst_trg = tst_trg[randomize]
        tst_lbls = tst_lbls[randomize]

        trn_lbls = self._sft_crisp_lbl_(trn_lbls)
        tst_lbls = self._sft_crisp_lbl_(tst_lbls)
        
        _dct_ = {"trn_fts": trn_fts, 
            "trn_trg": trn_trg, 
            "trn_lbls": trn_lbls,
            "vld_fts": tst_fts, 
            "vld_trg": tst_trg, 
            "vld_lbls": tst_lbls,
            "tst_fts": tst_fts, 
            "tst_trg": tst_trg, 
            "tst_lbls": tst_lbls,
            }

        return _dct_

    def load_3d_dataset(self, target, rows):

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
            if rows == 28:          
                face_path_train = "../../GANs_models/cremad/test_cremad.pkl"
                face_path_test = "../../GANs_models/cremad/test_cremad.pkl"
            elif rows == 112:
                face_path_train = "../../GANs_models/cremad/test_cremad_big.pkl"
                face_path_test = "../../GANs_models/cremad/test_cremad_big.pkl" 

            _dt_ = self.load_obj(face_path_train)
      
            trn_fts= _dt_["specs_train"]
            trn_trg = _dt_["face_train"]
            trn_lbls = _dt_["train_lbls"]

            trn_trg = _norm_(trn_trg)

            _dt_ = self.load_obj(face_path_test)

            tst_fts = _dt_["specs_train"]
            tst_trg = _dt_["face_train"]
            tst_lbls= _dt_["train_lbls"]
            tst_trg = _norm_(tst_trg)
            
            randomize = np.arange(len(trn_fts))
            np.random.shuffle(randomize)
            trn_fts = trn_fts[randomize]
            trn_trg = trn_trg[randomize]
            trn_lbls = trn_lbls[randomize]

            randomize = np.arange(len(tst_fts))
            np.random.shuffle(randomize)
            tst_fts = tst_fts[randomize]
            tst_trg = tst_trg[randomize]
            tst_lbls = tst_lbls[randomize]

            trn_lbls = to_categorical(trn_lbls)
            trn_lbls = trn_lbls[:, 1:]

            tst_lbls = to_categorical(tst_lbls)
            tst_lbls = tst_lbls[:, 1:]

            trn_lbls = self._sft_crisp_lbl_(trn_lbls)
            tst_lbls = self._sft_crisp_lbl_(tst_lbls)

            _dct_ = {"trn_fts": trn_fts, 
                "trn_trg": trn_trg, 
                "trn_lbls": trn_lbls,
                "vld_fts": trn_fts, 
                "vld_trg": trn_trg, 
                "vld_lbls": trn_lbls,
                "tst_fts": tst_fts, 
                "tst_trg": tst_trg, 
                "tst_lbls": tst_lbls,
                }

            return _dct_

   
    def load_3d_dataset_v2(self, target, rows):
           
        path = "../models/2dCNN_features.pkl"

        _dt_ = self.load_obj(path)

        trn_fts= _dt_["trn_fts"]
        trn_trg = _dt_["trn_trg"]
        trn_lbls = _dt_["trn_lbls"]

        tst_fts = _dt_["tst_fts"]
        tst_trg = _dt_["trn_trg"]
        tst_lbls= _dt_["trn_lbls"]

        trn_fts = _norm_(trn_fts)
        tst_fts = _norm_(tst_fts)
                
        randomize = np.arange(len(trn_fts))
        np.random.shuffle(randomize)
        trn_fts = trn_fts[randomize]
        trn_trg = trn_trg[randomize]
        trn_lbls = trn_lbls[randomize]

        randomize = np.arange(len(tst_fts))
        np.random.shuffle(randomize)
        tst_fts = tst_fts[randomize]
        tst_trg = tst_trg[randomize]
        tst_lbls = tst_lbls[randomize]

        _dct_ = {"trn_fts": trn_fts, 
            "trn_trg": trn_trg, 
            "trn_lbls": trn_lbls,
            "vld_fts": trn_fts, 
            "vld_trg": trn_trg, 
            "vld_lbls": trn_lbls,
            "tst_fts": tst_fts, 
            "tst_trg": tst_trg, 
            "tst_lbls": tst_lbls,
            }

        return _dct_

                               

        
