from ignite.metrics import PSNR, SSIM, MeanSquaredError, MeanAbsoluteError
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *

import os
import numpy as np
from PIL import Image
import torch


# resizes a 1d numpy array to an arbitrary size
def resize(img, size):

  img = img.astype('float32')
  img = torch.tensor(img)
  img = img.unsqueeze(0)

  transform = T.Resize(size, antialias=True)
  resized_img = transform(img)

  return resized_img


# convert numpy aray to tensor and unsqueeze it
def convert_to_tensor(img):

  img = img.astype('float32')
  img = torch.tensor(img)
  img = img.unsqueeze(0)

  return img


# normalized an image
def norm(x):
    if np.amax(x) > 0:
        x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
    return x


# fill nan values in image with 1024 and reshape the image
def fill_nan_reshape(x, img_size):
    x = np.nan_to_num(x, nan=-1024.0)
    x = np.reshape(x, (img_size, img_size))
    return x


# reshape, convert to tensor and unsqueeze the input numpy array 
def pre_process(img, img_size):

    img = np.reshape(img, (1, img_size, img_size))
    img = torch.tensor(img)
    img = img.unsqueeze(0)

    return img


# create default evaluator for doctests

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)


def get_ssim(diff_outs, targets):

    metric = SSIM(data_range=1.0)
    metric.attach(default_evaluator, 'ssim')

    sum_ssims = 0
    all_ssims = []

    for (diff_out, target) in zip(diff_outs, targets):

        state = default_evaluator.run([[diff_out, target]])
        ssim_value = state.metrics['ssim']

        all_ssims.append(ssim_value)
        sum_ssims += ssim_value

    test_size = len(diff_outs)
    avg_ssim = sum_ssims / test_size

    all_ssims = np.array(all_ssims)
    max_ssim = all_ssims.max()
    argmax_ssim = all_ssims.argmax()

    return max_ssim, argmax_ssim, avg_ssim


def get_psnr(diff_outs, targets):

    metric = PSNR(data_range=1.0)
    metric.attach(default_evaluator, 'psnr')

    sum_psnrs = 0
    all_psnrs = []

    for (diff_out, target) in zip(diff_outs, targets):

        state = default_evaluator.run([[diff_out, target]])
        psnr_value = state.metrics['psnr']

        all_psnrs.append(psnr_value)
        sum_psnrs += psnr_value
    
    test_size = len(diff_outs)
    avg_psnr = sum_psnrs / test_size

    all_psnrs = np.array(all_psnrs)
    max_psnr = all_psnrs.max()
    argmax_psnr = all_psnrs.argmax()

    return max_psnr, argmax_psnr, avg_psnr


def get_mse(diff_outs, targets):

    metric = MeanSquaredError()
    metric.attach(default_evaluator, 'mse')

    sum_mses = 0
    all_mses = []

    for (diff_out, target) in zip(diff_outs, targets):

        state = default_evaluator.run([[diff_out, target]])
        mse_value = state.metrics['mse']

        all_mses.append(mse_value)
        sum_mses += mse_value

    test_size = len(diff_outs)
    avg_mse = sum_mses / test_size

    all_mses = np.array(all_mses)
    min_mse = all_mses.min()
    argmin_mse = all_mses.argmin()

    return min_mse, argmin_mse, avg_mse


def get_mae(diff_outs, targets):

    metric = MeanAbsoluteError()
    metric.attach(default_evaluator, 'mae')

    sum_maes = 0
    all_maes = []

    for (diff_out, target) in zip(diff_outs, targets):

        state = default_evaluator.run([[diff_out, target]])
        mae_value = state.metrics['mae']

        all_maes.append(mae_value)
        sum_maes += mae_value

    test_size = len(diff_outs)
    avg_mae = sum_maes / test_size

    all_maes = np.array(all_maes)
    min_mae = all_maes.min()
    argmin_mae = all_maes.argmin()

    return min_mae, argmin_mae, avg_mae


# save samples with best ssim, psnr, mse, mae in four seperate folders
# each folder contains numpy arrays of diffusion output and target and difusion output png file
def save_best_samples(sampler_type, DIREC, diff_outs, targets, argmax_ssim, argmax_psnr, argmin_mse, argmin_mae):

    best_ssim_folder_path = './current experiment/best ssim'
    best_psnr_folder_path ='./current experiment/best psnr'
    best_mse_folder_path ='./current experiment/best mse'
    best_mae_folder_path ='./current experiment/best mae'

    if not os.path.exists(best_ssim_folder_path):
        os.makedirs(best_ssim_folder_path)

    if not os.path.exists(best_psnr_folder_path):
        os.makedirs(best_psnr_folder_path)

    if not os.path.exists(best_mse_folder_path):
        os.makedirs(best_mse_folder_path)

    if not os.path.exists(best_mae_folder_path):
        os.makedirs(best_mae_folder_path)


    best_ssim_path = f'./current experiment/Train_Output_{sampler_type}/' + DIREC + f'/sample_{argmax_ssim}.png'
    best_psnr_path = f'./current experiment/Train_Output_{sampler_type}/' + DIREC + f'/sample_{argmax_psnr}.png'
    best_mse_path = f'./current experiment/Train_Output_{sampler_type}/' + DIREC + f'/sample_{argmin_mse}.png'
    best_mae_path = f'./current experiment/Train_Output_{sampler_type}/' + DIREC + f'/sample_{argmin_mae}.png'


    best_ssim_file_name = f'sample_{argmax_ssim}.png'
    best_psnr_file_name = f'sample_{argmax_psnr}.png'
    best_mse_file_name = f'sample_{argmin_mse}.png'
    best_mae_file_name = f'sample_{argmin_mae}.png'

    best_ssim_save_path = os.path.join(best_ssim_folder_path, best_ssim_file_name)
    best_ssim = Image.open(best_ssim_path)
    best_ssim.save(best_ssim_save_path)

    best_psnr_save_path = os.path.join(best_psnr_folder_path, best_psnr_file_name)
    best_psnr = Image.open(best_psnr_path)
    best_psnr.save(best_psnr_save_path)

    best_mse_save_path = os.path.join(best_mse_folder_path, best_mse_file_name)
    best_mse = Image.open(best_mse_path)
    best_mse.save(best_mse_save_path)

    best_mae_save_path = os.path.join(best_mae_folder_path, best_mae_file_name)
    best_mae = Image.open(best_mae_path)
    best_mae.save(best_mae_save_path)


    best_ssim_diff_out = diff_outs[argmax_ssim]
    best_psnr_diff_out = diff_outs[argmax_psnr]
    best_mse_diff_out = diff_outs[argmin_mse]
    best_mae_diff_out = diff_outs[argmin_mae]

    best_ssim_target = targets[argmax_ssim]
    best_psnr_target = targets[argmax_psnr]
    best_mse_target = targets[argmin_mse]
    best_mae_target = targets[argmin_mae]

    np.save(os.path.join(best_ssim_folder_path, f'diff_out_{argmax_ssim}'), best_ssim_diff_out)
    np.save(os.path.join(best_ssim_folder_path, f'target_{argmax_ssim}'), best_ssim_target)

    np.save(os.path.join(best_psnr_folder_path, f'diff_out_{argmax_psnr}'), best_psnr_diff_out)
    np.save(os.path.join(best_psnr_folder_path, f'target_{argmax_psnr}'), best_psnr_target)

    np.save(os.path.join(best_mse_folder_path, f'diff_out_{argmin_mse}'), best_mse_diff_out)
    np.save(os.path.join(best_mse_folder_path, f'target_{argmin_mse}'), best_mse_target)

    np.save(os.path.join(best_mae_folder_path, f'diff_out_{argmin_mae}'), best_mae_diff_out)
    np.save(os.path.join(best_mae_folder_path, f'target_{argmin_mae}'), best_mae_target)


# save average and best values of ssim, psnr, mse, mae in a text file
def save_metrics(sampler_type, avg_time, avg_ssim, avg_psnr, avg_mse, avg_mae, max_ssim, max_psnr, min_mse, min_mae):

    with open('./current experiment/metrics', 'w') as file:

        file.write(f'{sampler_type}\n')
        file.write(f'average time: {format(avg_time, ".2f")} seconds\n\n')

        file.write(f'average ssim: {format(avg_ssim, ".2f")}\n')
        file.write(f'average psnr: {format(avg_psnr, ".2f")}\n')
        file.write(f'average mse: {format(avg_mse, ".2f")}\n')
        file.write(f'average mae: {format(avg_mae, ".2f")}\n\n')

        file.write(f'max ssim: {format(max_ssim, ".2f")}\n')
        file.write(f'max psnr: {format(max_psnr, ".2f")}\n')
        file.write(f'min mse: {format(min_mse, ".2f")}\n')
        file.write(f'min mae: {format(min_mae, ".2f")}\n')