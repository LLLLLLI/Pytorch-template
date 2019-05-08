from model.model import Model
from data.dataloader import get_dataloader

import torch
from tools import AverageMeter, LogCollector

import logging
import tensorboard_logger as tb_logger
import argparse

def main():
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='', type=str, choices=['', ''])
    parser.add_argument('--feature_path', default='',
                        help='path to datasets')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='prefix of feature')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--iter_size', default=1, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--weight_decay', default=0.005, type=float)
    parser.add_argument('--grad_clip', default=1., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.001, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=38, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--storage_place', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    opt = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    train_loader, val_loader = temporal_conv_data.get_loaders(opt)

    # Construct the model
    model = VSE(opt)
    # model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7]).cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            #validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0

    

