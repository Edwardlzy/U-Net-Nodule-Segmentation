import numpy as np
import os

working_path = "..\\..\\datasets\\data_science_bowl_2017\\processed_LUNA\\subset"
train_X = np.zeros((1, 1, 512, 512))
train_Y = np.zeros((1, 1, 512, 512))
test_X = np.zeros((1, 1, 512, 512))
test_Y = np.zeros((1, 1, 512, 512))
num_examples = 0
test_examples = 0

for i in range(5):
    cur_dir = working_path + str(i)
    cur_X = np.load(os.path.join(cur_dir, "trainImages.npy"))
    cur_Y = np.load(os.path.join(cur_dir, "trainMasks.npy"))
    cur_test_X = np.load(os.path.join(cur_dir, "testImages.npy"))
    cur_test_Y = np.load(os.path.join(cur_dir, "testMasks.npy"))

    train_X = np.concatenate((train_X, cur_X), axis=0)
    train_Y = np.concatenate((train_Y, cur_Y), axis=0)
    test_X = np.concatenate((test_X, cur_test_X), axis=0)
    test_Y = np.concatenate((test_Y, cur_test_Y), axis=0)
    num_examples += len(cur_X)
    print('num_examples =', num_examples)
    test_examples += len(cur_test_X)
    print('test_examples =', test_examples)

save_path = "..\\..\\datasets\\data_science_bowl_2017\\processed_LUNA"
np.save(os.path.join(save_path, 'trainImages.npy'), train_X[1:])
np.save(os.path.join(save_path, 'trainMasks.npy'), train_Y[1:])
np.save(os.path.join(save_path, 'testImages.npy'), test_X[1:])
np.save(os.path.join(save_path, 'testMasks.npy'), test_Y[1:])