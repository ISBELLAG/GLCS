
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from optuna.trial import TrialState
from torch import device
import Config as config
from DataSet_loader import *
import logging
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import *
from Train_one_epoch import train_one_epoch
from Val_one_epoch import val_one_epoch
import numpy as np
from nets.CGAN import UNet,resnet50
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state,save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth'.format(model)
        # filename = save_path + '/' + \
        #            'best_model-{}.pth'.format(model) + str(trial_number)
        # print(model.state_dict())
    else:
        filename = save_path + '/' + \
               'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)
    # print(state)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)

def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):

    train_tf = transforms.Compose([RandomGenerator1to1(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    dataset = ImageTo1to1(config.sup_dataset, config.unsup_dataset, train_tf, image_size=config.img_size)
    val_dataset = Image2D(config.val_dataset, val_tf, image_size=config.img_size)
    train_dataloader = DataLoader(dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  worker_init_fn=worker_init_fn,
                                  num_workers=16,
                                  drop_last=True,
                                  pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=16,
                            drop_last=False,
                            pin_memory=True)
    logger.info(config.model_type)


    if config.model_type == 'UNet':
        model = UNet(3, 1)
    else:
        raise TypeError('Please enter a valid name for the model type')

    # Generator 部分
    model = model.cuda()  # sup
    model_D = resnet50(1,True)
    model_D = model_D.cuda()  # GAN

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model_D = nn.DataParallel(model_D)

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    criterionG = nn.BCEWithLogitsLoss()
    criterionD = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_D = torch.optim.SGD(model_D.parameters(), lr=Dlr)


    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
        lr_d = CosineAnnealingWarmRestarts(optimizer_D, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None

    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1

    #################################################################

    for epoch in range(config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with label batch size : {}'.format(config.batch_size))
        train_loss, train_dice, train_lossD, train_lossG = train_one_epoch(train_dataloader, model, model_D, criterion,
                                                                        criterionG, criterionD, optimizer, optimizer_D,
                                                                         writer, epoch, lr_scheduler,lr_d,
                                                                        config.model_type, logger)
        # =============================================================
        #       evaluate on validation set
        # =============================================================
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = val_one_epoch(val_loader, model, criterion,
                                               optimizer, writer, epoch, lr_scheduler, config.model_type, logger)
        # ===============================================================
        #                      Save best model
        # ===============================================================

        dice = 0.5 * train_dice + 0.5 * val_dice
        if dice > max_dice:
            if epoch + 1 > 0:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,
                                                                                              dice))  # gai
                max_dice = dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': config.model_type,
                                 'state_dict': model.state_dict(),
                                 "D_state_dice": model_D.state_dict(),
                                 'model_D': model_D.state_dict(),
                                 'optimizer_D': optimizer_D.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'val_loss': val_loss,
                                 }, config.model_path)

        else:
            logger.info('\t Mean dice:{:.4f} does not increase,'
                        'the best is still: {:.4f} in epoch {}'.format(dice, max_dice, best_epoch))  # gai
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break
    return max_dice

if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)
    logger = logger_config(log_path=config.logger_path)
    model = main_loop(model_type=config.model_name, tensorboard=True)