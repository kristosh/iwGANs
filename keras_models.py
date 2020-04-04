from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge
from keras.layers import Reshape, Concatenate
from keras.layers.core import Activation, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D, ZeroPadding2D, Conv2D
from keras.layers.core import Flatten
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import LSTM

import pdb

#from skimage.transform import resize
from normalization import BatchNormGAN
#from keras import backend as K


img_rows = 28
img_cols = 28
in_ch = 3
batch_size = 100
img_shape = (img_rows, img_cols, in_ch)
latent_dim = 60


def generator_model_temporal():

    global batch_size
    #inputs = Input((in_ch, img_cols, img_rows)) 
    inputs = Input(shape=(170,))
    #e0 = BatchNormGAN()(inputs)
    #e1 = Flatten()(inputs)
    #e2 = Concatenate()([e1, input_conditional])   
    #e3 = BatchNormGAN()(e1)
    e4 = Dense(1024, activation="linear")(inputs)
    e4 = LeakyReLU(alpha=0.1)(e4)
    e5 = BatchNormGAN()(e4)

    e5 = Dense(1024, activation="linear")(e5)
    e5 = LeakyReLU(alpha=0.1)(e5)
    e5 = BatchNormGAN()(e5)

    e5 = Dense(1024, activation="linear")(e5)
    e5 = LeakyReLU(alpha=0.1)(e5)
    e5 = BatchNormGAN()(e5)

    e5 = Dense(1024, activation="linear")(e5)
    e5 = LeakyReLU(alpha=0.1)(e5)
    e5 = BatchNormGAN()(e5)

    e6 = Dense(512, activation="linear")(e5)
    e6 = LeakyReLU(alpha=0.1)(e6)
    e7 = BatchNormGAN()(e6)
    e8 = Dense(512, activation="linear")(e7)
    e9 = LeakyReLU(alpha=0.1)(e8)
    e10 = BatchNormGAN()(e9)
    e11 = Dense(3 * 28 *112, activation="relu")(e10)
    e12  = Reshape((3, 28, 112))(e11)
    e13 = BatchNormGAN()(e12)
    e14 = Activation('tanh')(e13)

    model = Model([inputs], output=e14)
    return model


def generator_model(input_dim):

    global batch_size
    inputs = Input((in_ch, img_cols, img_rows)) 
    input_conditional = Input(shape=(input_dim,))

    e0 = BatchNormGAN()(inputs)
    e1 = Flatten()(e0)
    e2 = Concatenate()([e1, input_conditional])   
    e3 = BatchNormGAN()(e2)
    e4 = Dense(1024, activation="linear")(e3)
    e4 = LeakyReLU(alpha=0.1)(e4)
    e5 = BatchNormGAN()(e4)

    e5 = Dense(1024, activation="linear")(e5)
    e5 = LeakyReLU(alpha=0.1)(e5)
    e5 = BatchNormGAN()(e5)

    e5 = Dense(1024, activation="linear")(e5)
    e5 = LeakyReLU(alpha=0.1)(e5)
    e5 = BatchNormGAN()(e5)

    e5 = Dense(1024, activation="linear")(e5)
    e5 = LeakyReLU(alpha=0.1)(e5)
    e5 = BatchNormGAN()(e5)

    e6 = Dense(512, activation="linear")(e5)
    e6 = LeakyReLU(alpha=0.1)(e6)
    e7 = BatchNormGAN()(e6)
    e8 = Dense(512, activation="linear")(e7)
    e9 = LeakyReLU(alpha=0.1)(e8)
    e10 = BatchNormGAN()(e9)
    e11 = Dense(3 * 28 *112, activation="relu")(e10)
    e12  = Reshape((3, 28, 112))(e11)
    e13 = BatchNormGAN()(e12)
    e14 = Activation('tanh')(e13)

    model = Model([inputs, input_conditional], output=e14)
    return model



def discriminator_model():
    """ return a (b, 1) logits"""
    model = Sequential()
    model.add(Convolution2D(64, 4, 4,border_mode='same',input_shape=(3, 28, 112)))
    model.add(BatchNormGAN())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(512, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(1024, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(512, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(512, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(512, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(512, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(256, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(256, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Activation('sigmoid'))
    #model.summary()
    return model


def lenet_classifier_model_face(nb_classes):
    # Snipped by Fabien Tanc - https://www.kaggle.com/ftence/keras-cnn-inspired-by-lenet-5
    # Replace with your favorite classifier...
    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=(3, 28, 28)))
    model.add(Conv2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    #model.summary()
    return model


def lenet_classifier_model_face_v2(nb_classes):
    # Snipped by Fabien Tanc - https://www.kaggle.com/ftence/keras-cnn-inspired-by-lenet-5
    # Replace with your favorite classifier...
    model = Sequential()
    model.add(Convolution2D(12, 5, 5, activation='linear', input_shape=(3,28,28), init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(25, 5, 5, activation='linear', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(512, activation='linear', init='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(512, activation='linear', init='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(256, activation='linear', init='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(128, activation='linear', init='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    #model.add(Dense(512, activation='linear', init='he_normal'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(nb_classes, activation='softmax', init='he_normal'))
    #model.summary()
    return model


def lenet_classifier_model(nb_classes):
    # Snipped by Fabien Tanc - https://www.kaggle.com/ftence/keras-cnn-inspired-by-lenet-5
    # Replace with your favorite classifier...
    model = Sequential()
    model.add(Convolution2D(12, 5, 5, activation='linear', input_shape=(3,28,112), init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(25, 5, 5, activation='linear', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(512, activation='linear', init='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(512, activation='linear', init='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(256, activation='linear', init='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(128, activation='linear', init='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    #model.add(Dense(512, activation='linear', init='he_normal'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(nb_classes, activation='softmax', init='he_normal'))
    #model.summary()
    return model


def lenet_classifier_model_v2(nb_classes):
    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=(3, 28, 112)))
    model.add(Conv2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.summary()
    return model


def generator_model_rav():

    global batch_size
    inputs = Input((3, img_cols, img_rows)) 
    input_conditional = Input(shape=(13,))
    e0 = BatchNormGAN()(inputs)
    e1 = Flatten()(e0)
    e2 = Concatenate()([e1, input_conditional])   
    e3 = BatchNormGAN()(e2)
    e4 = Dense(1024, activation="linear")(e3)
    e4 = LeakyReLU(alpha=0.1)(e4)
    e5 = BatchNormGAN()(e4)

    e5 = Dense(1024, activation="linear")(e5)
    e5 = LeakyReLU(alpha=0.1)(e5)
    e5 = BatchNormGAN()(e5)

    e5 = Dense(1024, activation="linear")(e5)
    e5 = LeakyReLU(alpha=0.1)(e5)
    e5 = BatchNormGAN()(e5)

    e5 = Dense(1024, activation="linear")(e5)
    e5 = LeakyReLU(alpha=0.1)(e5)
    e5 = BatchNormGAN()(e5)

    e6 = Dense(512, activation="linear")(e5)
    e6 = LeakyReLU(alpha=0.1)(e6)
    e7 = BatchNormGAN()(e6)
    e8 = Dense(512, activation="linear")(e7)
    e9 = LeakyReLU(alpha=0.1)(e8)
    e10 = BatchNormGAN()(e9)
    e11 = Dense(3 * 28 *112, activation="relu")(e10)
    e12  = Reshape((3, 28, 112))(e11)
    e13 = BatchNormGAN()(e12)
    e14 = Activation('tanh')(e13)

    model = Model([inputs, input_conditional], output=e14)
    return model


def discriminator_model_rav():
    """ return a (b, 1) logits"""
    model = Sequential()
    model.add(Convolution2D(64, 4, 4,border_mode='same',input_shape=(3, 28, 112)))
    model.add(BatchNormGAN())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(512, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(1024, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(512, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(512, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(512, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(512, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(256, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(Convolution2D(256, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1, 4, 4,border_mode='same'))
    model.add(BatchNormGAN())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Activation('sigmoid'))
    model.summary()

    return model

# def lenet_classifier_model_face_v2(nb_classes):


def lenet_classifier_model_rav(nb_classes):
    # Snipped by Fabien Tanc - https://www.kaggle.com/ftence/keras-cnn-inspired-by-lenet-5
    # Replace with your favorite classifier...
    model = Sequential()
    model.add(Convolution2D(12, 5, 5, activation='linear', input_shape=(3,28,112), init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(25, 5, 5, activation='linear', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(512, activation='linear', init='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(512, activation='linear', init='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(256, activation='linear', init='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(128, activation='linear', init='he_normal'))
    model.add(LeakyReLU(alpha=0.1))
    #model.add(Dense(512, activation='linear', init='he_normal'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(nb_classes, activation='softmax', init='he_normal'))
    model.summary()

    return model

def cnnModel(input_shape, nClasses):

    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
    
    return model


#==========================================
#==========================================
#=============WGANS models=================
def build_generator_old(latent_dim):


    inputs = Input((latent_dim,))

    e0 = Dense(128 * 7 * 7, activation="relu")(inputs)
    e1 = Reshape((7, 7, 128))(e0)
    e2 = UpSampling2D()(e1)
    
    e3 = Convolution2D(128, 4, 4, activation='linear',init='uniform', border_mode='same')(e2)
    e3 = BatchNormalization(momentum = 0.8)(e3)
    e3 =  Activation('relu')(e3)
    e3 = UpSampling2D()(e3)
    
    e4 = Convolution2D(64, 4, 4, activation='linear',init='uniform', border_mode='same')(e3)
    e4 = BatchNormalization(momentum = 0.8)(e4)
    e4 =  Activation('relu')(e4)
    e4 = UpSampling2D(size=(1, 2))(e4)

    e5 = Convolution2D(64, 4, 4, activation='linear',init='uniform', border_mode='same')(e4)
    e5 = BatchNormalization(momentum = 0.8)(e5)
    e5 =  Activation('relu')(e5)
    e5 = UpSampling2D(size=(1, 2))(e5)

    e6 = Convolution2D(3, 4, 4, activation='linear',init='uniform', border_mode='same')(e5)
    e6 = Activation("tanh")(e6)

    model = Model(input=inputs, output=e6)
    return model


def build_generator_face(latent_dim, channels):

    model = Sequential()
    model.add(Dense(128 * 7 * 7, activation="relu", input_shape=(None, latent_dim))) #input_dim
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    #model.add(UpSampling2D(size=(1, 2)))
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    #model.add(UpSampling2D(size=(1, 2)))
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2D(channels, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    mdl = Model(noise, output = img)

    return mdl


def build_generator_updated(latent_dim, channels):

    inputs = Input((in_ch, img_cols, img_rows)) 
    input_conditional = Input(shape=(latent_dim,))

    e0 = BatchNormGAN()(inputs)
    e1 = Flatten()(e0)
    e2 = Concatenate()([e1, input_conditional])  

    pdb.set_trace()
    model = Sequential()
    model.add(Dense(128 * 7 * 7, activation="relu" )) #input_dim
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    #model.add(UpSampling2D(size=(1, 2)))
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    #model.add(UpSampling2D(size=(1, 2)))
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2D(channels, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    mdl = Model(noise, output = img)

    return mdl


def build_generator(latent_dim, channels):

    model = Sequential()
    model.add(Dense(128 * 7 * 7, activation="relu", input_shape=(None, latent_dim))) #input_dim
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D(size=(1, 2)))
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D(size=(1, 2)))
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu")) 
    #

    # model.add(Conv2D(64, kernel_size=4, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation("relu"))

    # model.add(Conv2D(64, kernel_size=4, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation("relu"))

    model.add(Conv2D(channels, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))

    model.summary()
    #pdb.set_trace()
    noise = Input(shape=(latent_dim,))
    img = model(noise)

    mdl = Model(noise, output = img)

    return mdl


def build_critic(img_shape):

    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    # model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU(alpha=0.2))

    # model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())

    image = Input(shape=img_shape)
    features = model(image)
    
    fake = Dense(1, activation='linear', name='generation')(features)
    aux = Dense(6, activation='softmax', name='auxiliary')(features)

    return Model(input = [image], output=[fake, aux])


#=============================================
#============LSTM model=======================
#=============================================
def lstm_model(no_of_layers):
    
    model = Sequential()
    model.add(LSTM(units = 256, input_shape = (10, 256), return_sequences = False, dropout = 0.5))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(no_of_layers, activation='sigmoid'))
    #model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(6, activation='sigmoid'))

    print(model.summary())
    return model


#================================================
#============= My classifier models =============
#================================================
def my_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(3, 28,112)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    
    model.summary()

    return model


def lenet_classifier_model_face(nb_classes):
    # Snipped by Fabien Tanc - https://www.kaggle.com/ftence/keras-cnn-inspired-by-lenet-5
    # Replace with your favorite classifier...
    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=(3, 28, 28)))
    model.add(Conv2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    #model.summary()
    return model