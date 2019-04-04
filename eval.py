import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from model import Unet, Unet_3D
from dataloader import LungDataset, ToTensor
import numpy as np
import os
import cv2
# from train import random_crop_roi

# os.environ["CUDA_VISIBLE_DEVICES"]=""


def compute_dice_coef(outputs, masks):
    """ Compute the dice coefficient loss between the output and groundtruth mask. """
    outputs = torch.squeeze(outputs)
    masks = torch.squeeze(masks)
    intersection = torch.sum(outputs * masks)
    return (2 * intersection + 1e-5) / (torch.sum(outputs) + torch.sum(masks) + 1e-5)


def load_image_npy(path):
    """ Load a numpy array given its path. """
    image = np.transpose(np.load(path), (1, 2, 0))
    image_norm = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image_norm


def load_mask_npy(path):
    """ Load a mask numpy array given its path. """
    return np.transpose(np.load(path), (1, 2, 0))


# class Resize(object):
#     """ Resize the image in numpy format. """
#     def __init__(self, down_factor):
#         self.down_factor = down_factor

#     def __call__(self, imagedata):
#         if self.down_factor < 4:
#             return imagedata[::self.down_factor, ::self.down_factor]
#         else:
#             return imagedata[::self.down_factor, ::self.down_factor, ::int(self.down_factor/2)]


class Resize(object):
    """ Resize the image in numpy format. """
    def __init__(self, down_factor):
        self.down_factor = down_factor

    def __call__(self, imagedata):
        # if self.down_factor < 4:
        #     return imagedata[::self.down_factor, ::self.down_factor]
        # else:
        #     # return imagedata[::self.down_factor, ::self.down_factor, ::int(self.down_factor/2)]
        imagedata = np.array(imagedata)
        blurred_img = cv2.GaussianBlur(imagedata, (self.down_factor + 1, self.down_factor + 1), self.down_factor / 2.0)
        return blurred_img[::self.down_factor, ::self.down_factor, 96:-96]


class DatasetFolderWithFileName(datasets.DatasetFolder):
    """My DatasetFolder which returns image path (useful for debugging)"""
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


def get_3d_data_loader(opts):
    """ Return the dataloader for 3d nodule images. """
    transform = transforms.Compose([# Resize(4),
                                    transforms.ToTensor()])

    images_dataset = DatasetFolderWithFileName(opts.images_path, load_image_npy, ['npy'], transform)
    # lungmasks_dataset = datasets.DatasetFolder(opts.lungmasks_path, load_mask_npy, ['npy'], transform)
    masks_dataset = DatasetFolderWithFileName(opts.masks_path, load_mask_npy, ['npy'], transform)

    # images_loader = DataLoader(dataset=images_dataset, batch_size=opts.batch_size, shuffle=False, pin_memory=True)
    # # lungmasks_loader = DataLoader(dataset=lungmasks_dataset, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
    # masks_loader = DataLoader(dataset=masks_dataset, batch_size=opts.batch_size, shuffle=False, pin_memory=True)

    # return images_loader, lungmasks_loader, masks_loader
    return images_dataset, masks_dataset


def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    # return Variable(x)
    return x


def random_crop_roi(mask, shape=[64, 128, 128]):
    """ 
    Randomly crop out the ROI given the mask tensor. 
    Cropped shape = [64, 64, 64]
    Returns: [[z_min, z_high], 
              [y_min, y_high], 
              [x_min, x_high]]
    """
    roi = mask.nonzero()
    # if len(roi.shape) != 3:
    #     print('Invalid mask!')
    #     return None
    z_min, z_max = roi[:, 0].min(), roi[:, 0].max()
    y_min, y_max = roi[:, 1].min(), roi[:, 1].max()
    x_min, x_max = roi[:, 2].min(), roi[:, 2].max()

    z_range = z_max - z_min
    y_range = y_max - y_min
    x_range = x_max - x_min

    z_high = max(shape[0] - z_range, 1)
    y_high = max(shape[1] - y_range, 1)
    x_high = max(shape[2] - x_range, 1)

    z_split = torch.randint(z_high, (1,), dtype=torch.uint8).item()
    y_split = torch.randint(y_high, (1,), dtype=torch.uint8).item()
    x_split = torch.randint(x_high, (1,), dtype=torch.uint8).item()

    # Make sure the crop does not exceed the boundary.
    while z_min - z_split < 0 or z_max + (shape[0] - z_range - z_split) >= mask.shape[0]:
        z_split = torch.randint(z_high, (1,), dtype=torch.uint8).item()
    while y_min - y_split < 0 or y_max + (shape[1] - y_range - y_split) >= mask.shape[1]:
        y_split = torch.randint(y_high, (1,), dtype=torch.uint8).item()
    while x_min - x_split < 0 or x_max + (shape[2] - x_range - x_split) >= mask.shape[2]:
        x_split = torch.randint(x_high, (1,), dtype=torch.uint8).item()

    return torch.tensor([[z_min - z_split, z_max + (shape[0] - z_range - z_split)],
                         [y_min - y_split, y_max + (shape[1] - y_range - y_split)],
                         [x_min - x_split, x_max + (shape[2] - x_range - x_split)]])



def test(images_loader, masks_loader, opts):
    """ Evaluate the model on the test set. """
    model = torch.load(opts.ckpt_path)
    model.eval()
    if torch.cuda.is_available(): model = model.cuda()
    # images = iter(images_loader)
    # masks = iter(masks_loader)
    dice_coef = 0.0

    for i in range(len(images_loader)):
        # image = images.next()
        # mask = masks.next()
        image = images_loader[i]
        mask = masks_loader[i]
        path = image[2]
        image = torch.unsqueeze(image[0], 0)
        mask = torch.unsqueeze(mask[0], 0)
        if mask.max() == 0: 
            print('invalid mask')
            # iter_images.next()
            continue
        print('Processing image', path)
        # print('mask.shape =', mask.shape)
        crop_idx = random_crop_roi(mask[0])
        # print('crop_idx =', crop_idx)
        mask = mask[:, crop_idx[0,0]:crop_idx[0,1], crop_idx[1,0]:crop_idx[1,1], crop_idx[2,0]:crop_idx[2,1]]
        image = image[:, crop_idx[0,0]:crop_idx[0,1], crop_idx[1,0]:crop_idx[1,1], crop_idx[2,0]:crop_idx[2,1]]
        mask[mask > 0] = 1
        image = to_var(image).float().unsqueeze(dim=1)
        mask = to_var(mask).float().unsqueeze(dim=1)

        # if i < 1: continue
        # print('predicting image...')
        pred = model(image)

        cur_dice_coef = compute_dice_coef(pred, mask)
        print('cur_dice_coef =', cur_dice_coef)
        dice_coef += cur_dice_coef.detach().cpu().numpy()
        # pred_mask = torch.squeeze(pred).detach().cpu().numpy()
        # save_path = os.path.join('./eval_result', os.path.splitext(os.path.basename(path))[0] + '_pred.npy')
        # np.save(save_path, pred_mask)
        # print('saved', save_path)
        # image = torch.squeeze(image).cpu().numpy()
        # save_path = os.path.join('./eval_result', os.path.splitext(os.path.basename(path))[0] + '_img.npy')
        # np.save(save_path, image)
        # mask = torch.squeeze(mask).cpu().numpy()
        # save_path = os.path.join('./eval_result', os.path.splitext(os.path.basename(path))[0] + '_mask.npy')
        # np.save(save_path, mask)
        # print('saved', save_path)

        del image
        del mask
        del pred
        del cur_dice_coef
        # exit(0)

    dice_coef /= len(images_loader)
    print('dice_coef =', dice_coef)
    return dice_coef


def create_parser():
    """ 
    Creates a parser for command-line arguments. 
    """
    parser = argparse.ArgumentParser(description='PyTorch Unet Example')
    parser.add_argument('--images_path', type=str, default='D:\\code\\datasets\\data_science_bowl_2017\\processed_LUNA_full\\3d_data_test\\images')
    # parser.add_argument('--lungmasks_path', type=str, default='D:\\code\\datasets\\data_science_bowl_2017\\processed_LUNA_full\\3d_data_test\\lungmasks')
    parser.add_argument('--masks_path', type=str, default='D:\\code\\datasets\\data_science_bowl_2017\\processed_LUNA_full\\3d_data_test\\masks')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints_relaxed/best_ckpt.pth')

    return parser


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def main(opts):
    """
    Train the 3d Unet on 3d lung nodule dataset.
    """
    # images_loader, lungmasks_loader, masks_loader = get_3d_data_loader(opts)
    images_loader, masks_loader = get_3d_data_loader(opts)

    # if not os.path.isdir(opts.ckpt_path): os.mkdir(opts.ckpt_path)
    # test(images_loader, lungmasks_loader, masks_loader, opts)
    test(images_loader, masks_loader, opts)


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()
    print_opts(opts)
    main(opts)