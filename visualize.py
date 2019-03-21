import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


img_path = 'D:\\code\\datasets\\data_science_bowl_2017\\processed_LUNA_full\\3d_data_test\\images\\data\\images_0003_0017_img.npy'
pred_path = 'D:\\code\\datasets\\data_science_bowl_2017\\processed_LUNA_full\\3d_data_test\\images\\data\\images_0003_0017_pred.npy'
mask_path = 'D:\\code\\datasets\\data_science_bowl_2017\\processed_LUNA_full\\3d_data_test\\masks\\data\\masks_0003_0017.npy'
lungmask_path = 'D:\\code\\datasets\\data_science_bowl_2017\\processed_LUNA_full\\3d_data_test\\lungmasks\\data\\lungmask_0003_0017.npy'

imgs = np.load(img_path)    # (119, 512, 512)
masks = np.load(lungmask_path)
nodule_masks = np.load(mask_path)
preds = np.load(pred_path)

# Threshold the preds matrix
sorted_idx = np.unravel_index(np.argsort(preds, axis=None), preds.shape)
threshold = preds[sorted_idx[0][-1000], sorted_idx[1][-1000], sorted_idx[2][-1000]]
preds[preds < threshold] = 0

# imgs = imgs[::4, ::8, ::8]
# masks = masks[::4, ::8, ::8]
# nodule_masks = nodule_masks[::4, ::8, ::8]

print('nodule_masks.sum =', np.sum(nodule_masks))

print('Saving nodule masks...')
for i in range(len(nodule_masks)):
    cur = nodule_masks[i]
    if np.sum(cur) > 0:
        print('At i = {}, nodule present with area = {}!'.format(str(i), str(np.sum(cur))))
    scipy.misc.imsave('./0003_0017/nodule_'+str(i)+'.jpg', nodule_masks[i]*255)

print('Saving masks...')
for i in range(len(masks)):
    scipy.misc.imsave('./0003_0017/mask_'+str(i)+'.jpg', masks[i])

print('Saving original images...')
for i in range(len(imgs)):
    scipy.misc.imsave('./0003_0017/img_'+str(i)+'.jpg', imgs[i])

print('Saving predictions...')
for i in range(len(preds)):
    scipy.misc.imsave('./0003_0017/pred_'+str(i)+'.jpg', preds[i])

# for i in range(len(imgs)):
#     print ("image", i)
#     fig,ax = plt.subplots(2,2,figsize=[8,8])
#     ax[0,0].imshow(imgs[i],cmap='gray')
#     ax[0,1].imshow(masks[i],cmap='gray')
#     ax[1,0].imshow(imgs[i]*masks[i],cmap='gray')
#     ax[1,1].imshow(imgs[i]*nodule_masks[i],cmap='gray')
#     plt.show()
#     # raw_input("hit enter to cont : ")