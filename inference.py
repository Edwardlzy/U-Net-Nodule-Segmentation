import numpy as np
import os
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from LUNA_train_unet import get_unet, dice_coef, dice_coef_np

working_path = "..\\..\\datasets\\data_science_bowl_2017\\processed_LUNA\\"
test_img = np.load(os.path.join(working_path, 'testImages.npy')).astype(np.float32)
test_mask = np.load(os.path.join(working_path, 'testMasks.npy')).astype(np.float32)

mean = np.mean(test_img)
std = np.std(test_img)

test_img -= mean
test_img /= std

print('-'*30)
print('Creating and compiling model...')
print('-'*30)
model = get_unet()
model.load_weights('./unet.hdf5')

print('-'*30)
print('Predicting masks on test data...')
print('-'*30)

num_test = len(test_img)
imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
for i in range(num_test):
    imgs_mask_test[i] = model.predict(test_img[i].reshape((1, 1, 512, 512)), verbose=0)[0]
np.save('masksTestPredicted.npy', imgs_mask_test)

mean = 0.0
for i in range(num_test):
    mean += dice_coef_np(test_mask[i,0], imgs_mask_test[i,0])
mean/=num_test
print("Mean Dice Coeff : ",mean)