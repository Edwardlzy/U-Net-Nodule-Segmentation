from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from model import Unet
from dataloader import LungDataset, ToTensor
import numpy as np
import os


def dice_coef_loss(outputs, masks):
    """ Compute the dice coefficient loss between the output and groundtruth mask. """
    intersection = torch.sum(outputs * masks)
    return -(2 * intersection + 1) / (torch.sum(outputs) + torch.sum(masks) + 1)


def load_image_npy(path):
    """ Load a numpy array given its path. """
    image = np.transpose(np.load(path), (1, 2, 0))
    image_norm = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image_norm


def load_mask_npy(path):
    """ Load a mask numpy array given its path. """
    return np.transpose(np.load(path), (1, 2, 0))


def get_3d_data_loader():
    """ Return the dataloader for 3d nodule images. """
    transform = transforms.Compose([transforms.ToTensor()])

    images_path = 'D:\\code\\datasets\\data_science_bowl_2017\\processed_LUNA\\3d_data\\images'
    lungmasks_path = 'D:\\code\\datasets\\data_science_bowl_2017\\processed_LUNA\\3d_data\\lungmasks'
    masks_path = 'D:\\code\\datasets\\data_science_bowl_2017\\processed_LUNA\\3d_data\\masks'

    images_dataset = datasets.DatasetFolder(images_path, load_image_npy, ['npy'], transform)
    lungmasks_dataset = datasets.DatasetFolder(lungmasks_path, load_mask_npy, ['npy'], transform)
    masks_dataset = datasets.DatasetFolder(masks_path, load_mask_npy, ['npy'], transform)

    # images_loader = DataLoader(dataset=images_dataset, batch_size=1, shuffle=True, num_workers=opts.num_workers)
    # test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    return train_dloader, test_dloader


# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (image, mask) in enumerate(train_loader):
#         print('image.shape =', image.shape, 'mask.shape =', mask.shape)
#         image, mask = image.to(device), mask.to(device)
#         optimizer.zero_grad()
#         output = model(image)
#         loss = dice_coef_loss(output, mask)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(image), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))


def train(args, model, device, training_images, training_masks, optimizer, epoch):
    model.train()
    best_acc = 0.0

    for i in range(len(training_images) // 8):
        cur_batch_img = training_images[i * 8 : (i + 1) * 8]
        cur_batch_mask = training_masks[i * 8 : (i + 1) * 8]
        # cur_batch_img = training_images[i].reshape((1,1,512,512))
        # cur_batch_mask = training_masks[i].reshape((1,1,512,512))
        # print('image.shape =', cur_batch_img.shape, 'mask.shape =', cur_batch_mask.shape)
        image, mask = cur_batch_img.to(device), cur_batch_mask.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = dice_coef_loss(output, mask)
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(image), len(training_images),
                100. * i * args.batch_size / len(training_images), loss.item()))
        if -loss.item() > best_acc:
            if os.path.isfile('ckpt_' + str(epoch) + '_' + str(best_acc) + '.pt'):
                os.remove('ckpt_' + str(epoch) + '_' + str(best_acc) + '.pt')
            best_acc = -loss.item()
            torch.save(model.state_dict(), 'ckpt_' + str(epoch) + '_' + str(best_acc) + '.pt')


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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Unet Example')
    parser.add_argument('--training_data', type=str, default='../datasets/data_science_bowl_2017/processed_LUNA/trainImages.npy')
    parser.add_argument('--training_mask', type=str, default='../datasets/data_science_bowl_2017/processed_LUNA/trainMasks.npy')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--checkpoint_path', type=str, default='./ckpt.pt')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0e-5, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # train_dataset = LungDataset(args.training_data, args.training_mask, 
    #                             transform=transforms.Compose([ToTensor()]))
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Unet().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    training_images = np.load(args.training_data)
    mean = np.mean(training_images)
    std = np.std(training_images)
    training_images = (training_images - mean) / std
    training_masks = np.load(args.training_mask)
    indices = np.arange(len(training_images))
    print('Start training...')
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(indices)
        cur_images = torch.from_numpy(np.array([(training_images[i] - np.min(training_images[i])) / (np.max(training_images[i]) - np.min(training_images[i])) for i in indices], dtype=np.float32))
        # cur_images = torch.from_numpy(np.array([training_images[i] for i in indices], dtype=np.float32))
        cur_masks = torch.from_numpy(np.array([training_masks[i] for i in indices], dtype=np.float32))
        train(args, model, device, cur_images, cur_masks, optimizer, epoch)
        # test(args, model, device, test_loader)


if __name__ == '__main__':
    main()