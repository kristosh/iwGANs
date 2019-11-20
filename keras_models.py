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

from skimage.transform import resize
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

def generator_model():

    global batch_size
    inputs = Input((in_ch, img_cols, img_rows)) 
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


def build_generator_iwGANs():

    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # model.add(UpSampling2D(size= (1, 2)))
    # model.add(Conv2D(64, kernel_size=4, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation("relu"))

    # model.add(Conv2D(64, kernel_size=4, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation("relu"))

    # model.add(UpSampling2D(size= (1, 2)))
    # model.add(Conv2D(64, kernel_size=4, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation("relu"))

    # model.add(Conv2D(64, kernel_size=4, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation("relu"))

    model.add(Conv2D(3, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))

    model.summary()
    #pdb.set_trace()
    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

def build_critic_iwGANs():

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
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    img = Input(shape=img_shape)
    
    features = model(img)

    fake = Dense(1, activation='linear', name='generation')(features)
    aux = Dense(6, activation='softmax', name='auxiliary')(features)

    return Model(img, output = [fake, aux])

def build_classifier_iwGANs():

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
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    img = Input(shape=img_shape)
    
    features = model(img)

    aux = Dense(6, activation='softmax', name='auxiliary')(features)

    return Model(img, output = aux)


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