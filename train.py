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
from tensorboardX import SummaryWriter

def dice_coef_loss(outputs, masks):
    """ Compute the dice coefficient loss between the output and groundtruth mask. """
    outputs = torch.squeeze(outputs)
    masks = torch.squeeze(masks)
    intersection = torch.sum(outputs * masks)
    return -(2 * intersection + 1e-5) / (torch.sum(outputs) + torch.sum(masks) + 1e-5)


def ce_loss(outputs, masks):
    """ Compute the cross entropy loss. """
    outputs = torch.squeeze(outputs)
    masks = torch.squeeze(masks)
    # total = masks.shape[0] * masks.shape[1] * masks.shape[2]
    # print('masks =', masks)
    # print('outputs =', outputs)
    # print('intermediate =', masks * torch.log2(outputs))
    intermediate = masks * torch.log2(outputs) + (1 - masks) * torch.log2(1 - outputs)
    nan_sum = torch.sum(torch.isnan(intermediate)).float()
    # print('nan num =', nan_sum)
    # intermediate[torch.isnan(intermediate)] = 0
    # loss = -1 * torch.sum(intermediate) / (total - nan_sum)
    loss = -1 * torch.sum(intermediate)
    return 0.1 * loss.float() + 10 * dice_coef_loss(outputs, masks)
    # return loss.float()


def load_image_npy(path):
    """ Load a numpy array given its path. """
    image = np.transpose(np.load(path), (1, 2, 0))
    image_norm = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image_norm


def load_mask_npy(path):
    """ Load a mask numpy array given its path. """
    return np.transpose(np.load(path), (1, 2, 0))


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


# class ToNumpy(object):
#     """ Converts the imagedata into numpy array. """
#     def __call__(self, imagedata):
#         img = np.array(imagedata)
#         return np.transpose(img, (2, 0, 1))


def get_3d_data_loader(opts):
    """ Return the dataloader for 3d nodule images. """
    transform = transforms.Compose([# ToNumpy()
                                    # Resize(4),
                                    transforms.ToTensor()
                                    ])

    images_dataset = datasets.DatasetFolder(opts.images_path, load_image_npy, ['npy'], transform)
    # lungmasks_dataset = datasets.DatasetFolder(opts.lungmasks_path, load_mask_npy, ['npy'], transform)
    masks_dataset = datasets.DatasetFolder(opts.masks_path, load_mask_npy, ['npy'], transform)

    images_loader = DataLoader(dataset=images_dataset, batch_size=opts.batch_size, shuffle=True)
    # lungmasks_loader = DataLoader(dataset=lungmasks_dataset, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
    masks_loader = DataLoader(dataset=masks_dataset, batch_size=opts.batch_size, shuffle=True)

    # return images_loader, lungmasks_loader, masks_loader
    return images_loader, masks_loader


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

# def train(images_loader, lungmasks_loader, masks_loader, opts):
def train(images_loader, masks_loader, opts):
    """
    Train the 3D Unet on the given datasets.
    """
    # Initialize or load the model
    save_path = os.path.join(opts.checkpoint_dir, 'best_ckpt.pth')
    if os.path.exists(save_path): 
        print('Loading from', save_path)
        model = torch.load(save_path)
        save_path = os.path.join(opts.checkpoint_dir, 'best_ckpt.pth')
    else: model = Unet_3D()
    if torch.cuda.is_available(): model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opts.lr)
    decay_steps = [opts.decay_step * (i + 1) for i in range(opts.epochs // opts.decay_step)]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, decay_steps, gamma=opts.decay_factor)

    iter_per_epoch = len(images_loader) // opts.batch_size
    best_acc = 0.0
    # criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=opts.checkpoint_dir)

    for epoch in range(opts.epochs):
        # Make sure the images and masks match.
        seed = np.random.randint(opts.epochs)
        torch.manual_seed(seed)
        iter_images = iter(images_loader)
        torch.manual_seed(seed)
        # iter_lungmasks = iter(lungmasks_loader)
        torch.manual_seed(seed)
        iter_masks = iter(masks_loader)

        batch_dice_coeff = 0.0

        for iteration in range(iter_per_epoch):
            cur_mask = iter_masks.next()[0]  # (1,256,512,512)
            if cur_mask.max() == 0: continue
            cur_mask[cur_mask > 0] = 1
            # if torch.sum(cur_mask) == 0:
            #     print('Invalid mask, continue')
            #     continue
            cur_image = iter_images.next()[0]

            # Randomly Crop out 
            crop_idx = random_crop_roi(cur_mask[0])
            cur_mask = cur_mask[:, crop_idx[0,0]:crop_idx[0,1], crop_idx[1,0]:crop_idx[1,1], crop_idx[2,0]:crop_idx[2,1]]
            cur_image = cur_image[:, crop_idx[0,0]:crop_idx[0,1], crop_idx[1,0]:crop_idx[1,1], crop_idx[2,0]:crop_idx[2,1]]
            image = to_var(cur_image).float().unsqueeze(dim=1)
            # lungmask = to_var(iter_lungmasks.next()[0]).float().unsqueeze(dim=1)
            mask = to_var(cur_mask).float().unsqueeze(dim=1)
            # train_X = image * lungmask                
            train_X = image

            # Training
            optimizer.zero_grad()
            pred = model(train_X)
            dice_coef = -dice_coef_loss(pred, mask)
            batch_dice_coeff += dice_coef.detach().cpu().numpy()
            loss = ce_loss(pred, mask)
            loss.backward()
            optimizer.step()

            if iteration % opts.log_interval == 0:
                writer.add_scalar('Train/dice_coef', dice_coef.item(), epoch * iter_per_epoch + iteration)
                writer.add_scalar('Train/loss', loss.item(), epoch * iter_per_epoch + iteration)

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                epoch, iteration * len(image), len(images_loader),
                100. * iteration * opts.batch_size / len(images_loader), loss.item()))

        batch_dice_coeff /= iter_per_epoch
        writer.add_scalar('Train/training_set_dice_coef', batch_dice_coeff.item(), epoch)
        writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)
        print('epoch =', epoch, 'dice_coef =', batch_dice_coeff)
        if batch_dice_coeff.item() > best_acc:
            best_acc = batch_dice_coeff.item()
            print('Saving model with dice coefficient =', best_acc)
            torch.save(model, save_path)

        lr_scheduler.step()

    writer.close()
    print('Finished')


def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)



# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
#             pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))


def create_parser():
    """ 
    Creates a parser for command-line arguments. 
    """
    parser = argparse.ArgumentParser(description='PyTorch Unet Example')
    parser.add_argument('--images_path', type=str, default='D:\\code\\datasets\\data_science_bowl_2017\\processed_LUNA_full\\3d_data\\images')
    # parser.add_argument('--lungmasks_path', type=str, default='D:\\code\\datasets\\data_science_bowl_2017\\processed_LUNA_full\\3d_data\\lungmasks')
    parser.add_argument('--masks_path', type=str, default='D:\\code\\datasets\\data_science_bowl_2017\\processed_LUNA_full\\3d_data\\masks')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_relaxed/')
    # parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--decay_step', type=int, default=20, help='Number of steps before decaying.')
    parser.add_argument('--decay_factor', type=float, default=0.2, help='The factor to be multiplied to the learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

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

    if not os.path.isdir(opts.checkpoint_dir): os.mkdir(opts.checkpoint_dir)
    # train(images_loader, lungmasks_loader, masks_loader, opts)
    train(images_loader, masks_loader, opts)

    # # Training settings
    
    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    # device = torch.device("cuda" if use_cuda else "cpu")

    # # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # # train_dataset = LungDataset(args.training_data, args.training_mask, 
    # #                             transform=transforms.Compose([ToTensor()]))
    # # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # # train_loader = torch.utils.data.DataLoader(
    # #     datasets.MNIST('../data', train=True, download=True,
    # #                    transform=transforms.Compose([
    # #                        transforms.ToTensor(),
    # #                        transforms.Normalize((0.1307,), (0.3081,))
    # #                    ])),
    # #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # # test_loader = torch.utils.data.DataLoader(
    # #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    # #                        transforms.ToTensor(),
    # #                        transforms.Normalize((0.1307,), (0.3081,))
    # #                    ])),
    # #     batch_size=args.test_batch_size, shuffle=True, **kwargs)


    # model = Unet().to(device)
    # # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training_images = np.load(args.training_data)
    # mean = np.mean(training_images)
    # std = np.std(training_images)
    # training_images = (training_images - mean) / std
    # training_masks = np.load(args.training_mask)
    # indices = np.arange(len(training_images))
    # print('Start training...')
    # for epoch in range(1, args.epochs + 1):
    #     np.random.shuffle(indices)
    #     cur_images = torch.from_numpy(np.array([(training_images[i] - np.min(training_images[i])) / (np.max(training_images[i]) - np.min(training_images[i])) for i in indices], dtype=np.float32))
    #     # cur_images = torch.from_numpy(np.array([training_images[i] for i in indices], dtype=np.float32))
    #     cur_masks = torch.from_numpy(np.array([training_masks[i] for i in indices], dtype=np.float32))
    #     train(args, model, device, cur_images, cur_masks, optimizer, epoch)
    #     # test(args, model, device, test_loader)


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()
    print_opts(opts)
    main(opts)