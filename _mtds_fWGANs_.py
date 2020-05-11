import cv2
import numpy as np
import math
import my_iwgans
import pdb


class MTDS():

    def __init__(self):
        self.temp = False
        self._rng = 75000


    def _cmb_ten_fr_(self, images):
        
        #images = images * 127.5 + 127.5
        return images.swapaxes(2,1).reshape((64, 28, 280, 3))

    # def _cmb_ten_fr_(self, images):
    #     #images = images * 127.5 + 127.5
    #     pdb.set_trace()
    #     return images.swapaxes(2,1).reshape((images[0], 
    #         images[2],
    #         images[1]*images[3], 
    #         images[4]))



    def _comb_imgs_(self, generated_images):

        generated_images =  np.transpose(generated_images , (0, 3, 1, 2))

        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[2:]
        image = np.zeros((3,(height+30)*shape[0], (width+30)*shape[1]),dtype=generated_images.dtype)

        for index, img in enumerate(generated_images):

            new_shape = (img.shape[0], 
                img.shape[1]+ 4, img.shape[2] + 4)
            img_ = np.zeros(new_shape)
            img_[:, 2:  2+img.shape[1], 
                2:  2+img.shape[2]] = img

            i = int(index/width)
            j = index % width
            image[:, i*new_shape[1]: (i+1)*new_shape[1],
                j*new_shape[2]: (j+1)*new_shape[2]] = img_[:, :, :]
        return image


    def _str_imgs_(self, images_db, filename):

        image = self._comb_imgs_(images_db)
        #pdb.set_trace()
        image = image * 127.5 + 127.5
        image = np.swapaxes(image, 0, 1)
        image = np.swapaxes(image, 1, 2)
        cv2.imwrite(filename,image)


    # Generate new samples using the trained Generator.
    def _gen_datwGANs(self, _dct_, file_name, _obj_): 

        weight_name = "../../GANs_models/" \
            +_obj_.input_feats \
            +"_" + _obj_.db \
            +"_"+str(_obj_.latent_dim) \
            +"_"+str(_obj_.learning_param) \
            +"_"+str(_obj_.input_feats) \
            +"_"+_obj_.input_type \
            +"_" + _obj_.comment \
            +"_"+str(_obj_.img_rows) \
            +"_gen_noise_feats_" \
            + file_name+"_"

        # weight_name = "../../GANs_models/"+\
        #     "3dCNN_creamad_38_0.0001_"+\
        #     "3dCNN_without_source_"+\
        #     "tenMiddle_28_gen_noise_feats_face_"
        pdb.set_trace()
        _obj_.generator.load_weights(weight_name)    
        
        noise = np.random.normal(0, 1, (_dct_["trn_fts"].shape[0], _obj_._noizeD))
        
        if _obj_.input_to_G == True:
            noise = np.concatenate([_dct_["trn_fts"], 
                noise, 
                _dct_["trn_lbls"]], 
                axis = 1)
        else:
            noise = np.concatenate([
                noise, 
                _dct_["trn_lbls"]], 
                axis = 1)
                
        gen_train = _obj_.generator.predict([noise])

        if _obj_.target_mod == "face":

            gen_data = {"gen_train": gen_train,
                "lbls_train": _dct_["trn_lbls"]}

            stored_name = "../../GANs_models/generated_" \
                    +_obj_.input_feats \
                    +"_" + _obj_.db \
                    +"_"+str(_obj_.latent_dim) \
                    +"_"+str(_obj_.learning_param) \
                    +"_"+str(_obj_.input_feats) \
                    +"_"+_obj_.input_type \
                    +"_" + _obj_.comment \
                    +"_"+str(_obj_.img_rows) \
                    +"_gen_noise_feats_" \
                    + file_name+"_" \
                    +".pkl"
            
            pdb.set_trace()
            #self._str_imgs_(gen_train[:1600], "faces.png")
            _obj_.obj.store_obj(stored_name, gen_data)

        else:
            for index in range(0, 2):
                
                gen_data = {"gen_train": gen_train[self._rng*index:(self._rng+ self._rng*index)], 
                    "lbls_train": _dct_["tst_lbls"][self._rng*index:(self._rng+ self._rng*index)]}

                stored_name = "../../GANs_models/generated_" \
                    +_obj_.input_feats \
                    +"_" + _obj_.db \
                    +"_"+str(_obj_.latent_dim) \
                    +"_"+str(_obj_.learning_param) \
                    +"_"+str(_obj_.input_feats) \
                    +"_"+_obj_.input_type \
                    +"_" + _obj_.comment \
                    +"_"+str(_obj_.img_rows) \
                    +"_gen_noise_feats_" \
                    + file_name+"_" \
                    +".pkl"

                _obj_.obj.store_obj(stored_name, gen_data)
                gen_data = None  