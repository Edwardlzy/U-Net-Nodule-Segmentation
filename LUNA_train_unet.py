from __future__ import print_function

import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, concatenate, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.activations import relu
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

working_path = "..\\..\\datasets\\data_science_bowl_2017\\processed_LUNA\\"

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def merge(inputs, mode, concat_axis=-1):
    return concatenate(inputs, concat_axis)

class Unet(keras.Model):
    def __init__(self, input_channel=1, img_rows=512, img_cols=512):
        super(Unet, self).__init__(name='unet')
        self.input_channel = input_channel
        self.img_rows = img_rows
        self.img_cols = img_cols

    def call(self, inputs):
        # inputs = Input((1,img_rows, img_cols))
        print('inputs.shape =', inputs.shape)

        conv1 = Conv2D(32, 3, padding='same')(inputs)
        print('conv1.shape =', conv1.shape)
        # conv1 = Convolution2D(32, 3, 3, border_mode='same')(inputs)
        conv1 = relu(BatchNormalization()(conv1))
        print('after relu+batchnorm, conv1.shape =', conv1.shape)
        conv1 = Conv2D(32, 3, padding='same')(conv1)
        conv1 = relu(BatchNormalization()(conv1))
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, 3, padding='same')(pool1)
        conv2 = relu(BatchNormalization()(conv2))
        conv2 = Conv2D(64, 3, padding='same')(conv2)
        conv2 = relu(BatchNormalization()(conv2))
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, padding='same')(pool2)
        conv3 = relu(BatchNormalization()(conv3))
        conv3 = Conv2D(128, 3, padding='same')(conv3)
        conv3 = relu(BatchNormalization()(conv3))
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, 3, padding='same')(pool3)
        conv4 = relu(BatchNormalization()(conv4))
        conv4 = Conv2D(256, 3, padding='same')(conv4)
        conv4 = relu(BatchNormalization()(conv4))
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, 3, padding='same')(pool4)
        conv5 = relu(BatchNormalization()(conv5))
        conv5 = Conv2D(512, 3, padding='same')(conv5)
        conv5 = relu(BatchNormalization()(conv5))

        upsample6 = UpSampling2D(size=(2, 2))
        up6 = merge([upsample6(conv5), conv4], mode='concat', concat_axis=1)
        conv6 = Conv2D(256, 3, padding='same')(up6)
        conv6 = relu(BatchNormalization()(conv6))
        conv6 = Conv2D(256, 3, padding='same')(conv6)
        conv6 = relu(BatchNormalization()(conv6))

        upsample7 = UpSampling2D(size=(2, 2))
        up7 = merge([upsample7(conv6), conv3], mode='concat', concat_axis=1)
        conv7 = Conv2D(128, 3, padding='same')(up7)
        conv7 = relu(BatchNormalization()(conv7))
        conv7 = Conv2D(128, 3, padding='same')(conv7)
        conv7 = relu(BatchNormalization()(conv7))

        upsample8 = UpSampling2D(size=(2, 2))
        up8 = merge([upsample8(conv7), conv2], mode='concat', concat_axis=1)
        conv8 = Conv2D(64, 3, padding='same')(up8)
        conv8 = relu(BatchNormalization()(conv8))
        conv8 = Conv2D(64, 3, padding='same')(conv8)
        conv8 = relu(BatchNormalization()(conv8))

        upsample9 = UpSampling2D(size=(2, 2))
        up9 = merge([upsample9(conv8), conv1], mode='concat', concat_axis=1)
        conv9 = Conv2D(32, 3, padding='same')(up9)
        conv9 = relu(BatchNormalization()(conv9))
        conv9 = Conv2D(32, 3, padding='same')(conv9)
        conv9 = relu(BatchNormalization()(conv9))

        # conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        return conv10


# def get_unet():
#     inputs = Input((1,img_rows, img_cols))

#     conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
#     conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
#     conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
#     conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#     conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
#     conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#     conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
#     conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

#     upsample6 = UpSampling2D(size=(2, 2))
#     up6 = merge([upsample6(conv5), conv4], mode='concat', concat_axis=1)
#     conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
#     conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

#     upsample7 = UpSampling2D(size=(2, 2))
#     up7 = merge([upsample7(conv6), conv3], mode='concat', concat_axis=1)
#     conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
#     conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

#     upsample8 = UpSampling2D(size=(2, 2))
#     up8 = merge([upsample8(conv7), conv2], mode='concat', concat_axis=1)
#     conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
#     conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

#     upsample9 = UpSampling2D(size=(2, 2))
#     up9 = merge([upsample9(conv8), conv1], mode='concat', concat_axis=1)
#     conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
#     conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

#     conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

#     model = Model(input=inputs, output=conv10)

#     model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

#     return model


def train_and_predict(use_existing):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train = np.load(working_path+"trainImages.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path+"trainMasks.npy").astype(np.float32)

    imgs_test = np.load(working_path+"testImages.npy").astype(np.float32)
    imgs_mask_test_true = np.load(working_path+"testMasks.npy").astype(np.float32)
    
    # mean = np.mean(imgs_train)  # mean for data centering
    # std = np.std(imgs_train)  # std for data normalization

    # imgs_train -= mean  # images should already be standardized, but just in case
    # imgs_train /= std

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    # model = get_unet()
    model = Unet()
    model.compile(optimizer=Adam(lr=1.0e-2), loss=dice_coef_loss, metrics=[dice_coef])
    # Saving weights to unet.hdf5 at checkpoints
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True, save_weights_only=True)
    #
    # Should we load existing weights? 
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        model.load_weights('./unet.hdf5')
        
    # 
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970 
    # I was able to run 20 epochs with a training set size of 320 and 
    # batch size of 2 in about an hour. I started getting reseasonable masks 
    # after about 3 hours of training. 
    #
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=20, verbose=1, shuffle=True,
              callbacks=[model_checkpoint])

    # loading best weights from training session
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('./unet.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    num_test = len(imgs_test)
    imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
    np.save('masksTestPredicted.npy', imgs_mask_test)
    mean = 0.0
    for i in range(num_test):
        mean+=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
    mean/=num_test
    print("Mean Dice Coeff : ",mean)

if __name__ == '__main__':
    train_and_predict(False)
