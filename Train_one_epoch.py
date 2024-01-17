import numpy as np
import torch.optim
import os
import time
import math
import random
from utils import *
import Config as config
from skimage.metrics import structural_similarity as ssim
from nets.CGAN import MS_SSIM
from torch.autograd import Variable
from torchvision import transforms
import warnings

from PIL import Image
from PIL import ImageEnhance
from itertools import cycle

from torch.nn.modules.loss import CrossEntropyLoss

warnings.filterwarnings("ignore")


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, dis_acc,  mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'IoU:{:.3f} '.format(iou)
    # string += '(Avg {:.4f}) '.format(average_iou)
    # string += 'real_acc:{:.4f} '.format(real_acc)
    # string += 'dis_acc:{:.4f} '.format(dis_acc)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary

##################################################################################
# ================================================================================
#          Train One Epoch
# ================================================================================
##################################################################################

def train_one_epoch(loader, model, model_D,
                  criterion, criterionG, criterionD, optimizer,  optimizer_D, writer, epoch, lr_scheduler,lr_d,
                  model_type, logger):
    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()
    time_sum, loss_sum = 0, 0
    loss_patch, loss_image_one = 0, 0
    dice_sum, iou_sum, lossD_sum,lossG_sum, acc_sum, dis_acc_fake, dis_acc_real, dacc, dacc_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dices = []



    for i, Sample in enumerate(loader,1):

        # Take variable and put them to GPU
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        images, label, unimage  = Sample['image'], Sample['label'],Sample['unimage']
        images, label  = images.cuda(), label.cuda()
        unimage = unimage.cuda()

        u_weak = dim_cat(unimage, 0, WeakAug1)  # [20,1,128,128]
        weak = dim_cat(unimage, 0, WeakAug)
        strong = dim_cat(unimage, StrongAugMean, StrongAugSigma)


        aug =u_weak.size(0)

        model.train()
        model_D.eval()
        inputs = interleave(torch.cat((images, u_weak, weak)), 2 * aug + config.batch_size).cuda()
        inputs = inputs.float()
        output = model(inputs)
        output = de_interleave(output, 2 * aug + config.batch_size)
        sup_x = output[:config.batch_size]
        unlabel_u_w1, unlabel_u_w2 = output[config.batch_size:].chunk(2)
        del inputs, output
        torch.cuda.empty_cache()

        """SUP part"""
        loss_sup = criterion(sup_x, label.float())
        # print("loss_sup", loss_sup)
        """loss weak part"""
    #===============================================
        loss_w = 0
        for idx in range(aug):
            w1 = unlabel_u_w1[idx, :, :, :]
            w2 = unlabel_u_w2[idx, :, :, :]
            lossw = ((w1 - w2) ** 2).mean()
            # print(idx, lossw)
            loss_w += lossw
            loss_w = loss_w.mean()
        # print("loss_w", loss_w)
        loss = sup_ratio * loss_sup + (1 - sup_ratio ) * loss_w

        if model.training:
            # #梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # """ GAN part """
        real_label = torch.ones(aug, 1).clone().detach().cuda()  # 1
        fake_label = torch.zeros(aug, 1).clone().detach().cuda()  # 0

        model.eval()
        model_D.train()

        with torch.no_grad():
            inputs = interleave(torch.cat((weak, strong)), 2 * aug).cuda()
            output = model(inputs).detach()      #gen   detach()截断梯度
            output = de_interleave(output, 2 * aug)
            unlabel_u_w2, unlabel_u_s = output.chunk(2)



        for patch_i in range(aug):

            pi_weak_list = cut_image(unlabel_u_w2[patch_i, :, :, :], patch_num=5)
            pi_strong_list = cut_image(unlabel_u_s[patch_i, :, :, :], patch_num=5)
            #计算loss
            Mssim = MS_SSIM()
            for pi_list in range(len(pi_weak_list)-1):
                weak_mssim = Mssim(np.array(pi_weak_list[pi_list]), np.array(pi_weak_list[pi_list+1])).mean()
                strong_mssim = Mssim(np.array(pi_strong_list[pi_list]), np.array(pi_strong_list[pi_list+1])).mean()
                loss_patch_one = (weak_mssim - strong_mssim) ** 2
                loss_image_one += loss_patch_one     #one image --->all patch  loss
            loss_patch += loss_image_one  #all batch loss

        true_w = unlabel_u_w2.size(0)
        true_s = unlabel_u_s.size(0)

        pred_w1 = model_D(unlabel_u_w2)
        loss_r1 = criterionD(pred_w1.detach(), real_label)



        pred_s1 = model_D(unlabel_u_s)
        loss_f1 = criterionD(pred_s1.detach(), fake_label)

        loss_D = loss_r1 + loss_f1


        optimizer_D.zero_grad()
        loss_D.requires_grad=True
        loss_D.backward()
        optimizer_D.step()

        # ----------------
        #  Train Generator
        # ----------------

        model_D.eval()
        model.train()
        g_s1 = model(strong)
        with torch.no_grad():
            preds1 = model_D(g_s1)
        loss_G = criterionG(preds1, fake_label)

        loss_G_all = gan_ratio * loss_patch + (1 - gan_ratio) * loss_G


        if epoch % 20 == 0:
            optimizer.zero_grad()
            loss_G_all.requires_grad = True
            loss_G_all.backward()
            optimizer.step()


        # train_iou = iou_on_batch(masks, outputs)
        # train_dice = criterion._show_dice(outputs, masks.float())
        # train_iou = iou_on_batch(label, sup_x)
        train_dice = criterion._show_dice(sup_x, label.float())
        batch_time = time.time() - end


        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path + str(epoch) + '/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images, label, sup_x, vis_path)
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * loss
        # iou_sum += len(images) * train_iou
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice
        # lossD_sum += len(unimage) * loss_D
        # lossG_sum += len(unimage) * loss_G_all

        if i == len(loader):
            average_loss = loss_sum / ((config.batch_size * (i - 1) + len(images)))
            average_lossD = lossD_sum / ((config.batch_size * (i - 1) + len(unimage)))
            average_lossG = lossG_sum / ((config.batch_size * (i - 1) + len(unimage)))
            average_time = time_sum / (config.batch_size * (i - 1) + len(images))
            # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            # train_iou_average = iou_sum / (config.batch_size * (i - 1) + len(images))
            train_dice_avg = dice_sum / ((config.batch_size * (i - 1) + len(images)))
        else:
            average_loss = loss_sum / ((i * config.batch_size))
            average_lossD = lossD_sum / ((i * config.batch_size))
            average_lossG = lossG_sum / ((i * config.batch_size))
            average_time = time_sum / ((i * config.batch_size))
            # train_iou_average = iou_sum / ((i * config.batch_size))
            # train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / ((i * config.batch_size))


        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), loss, loss_name, batch_time,
                          average_loss, average_time, 0, 0,
                          train_dice, train_dice_avg, 0, 0,  0, logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups), logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, loss.item(), step)
            # plot metrics in tensorboard
            # writer.add_scalar(logging_mode + '_iou', train_iou, step)
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)
        del loss
        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()
        lr_d.step()
    # if epoch + 1 > 10: # Plateau
    #     if lr_scheduler is not None:
    #         lr_scheduler.step(train_dice_avg)

    return average_loss, train_dice_avg, average_lossD, average_lossG

