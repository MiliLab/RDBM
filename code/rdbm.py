import copy
import glob
import math
import os
import random
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
import torchvision.transforms as transforms

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from PIL import Image
import time
from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm 
import copy 
 
ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_noise', 'pred_x_start']) 

def set_seed(SEED): 
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class RDBM(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size = 256,
        objective = 'pred_x_start', 
        sampling_type = 'pred_x_start',
        timesteps=100, 
        sampling_timesteps=10,  
        condition=True, 
    ):
        super().__init__()
        
        assert objective in ['pred_noise','pred_x_start']
        assert sampling_type in ['pred_noise','pred_x_start']
        
        self.model = model
        self.channels = self.model.channels
        self.image_size = image_size
        self.condition = condition
        self.objective = objective
        self.sampling_type = sampling_type
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps

        lamb = 1e-4
        theta_start = 0.0001
        theta_end = 0.02
        thetas = betas_for_alpha_bar(timesteps)
        thetas_cumsum_0_to_t = thetas.cumsum(dim=0)
        thetas_cumsum_0_to_T = thetas_cumsum_0_to_t[-1]
        thetas_cumsum_t_to_T = thetas_cumsum_0_to_T - thetas_cumsum_0_to_t

        sinh_thetas_cumsum_0_to_t = torch.sinh(thetas_cumsum_0_to_t)
        sinh_thetas_cumsum_0_to_T = torch.sinh(thetas_cumsum_0_to_T)
        sinh_thetas_cumsum_t_to_T = torch.sinh(thetas_cumsum_t_to_T)
        
        Theta = sinh_thetas_cumsum_t_to_T / (sinh_thetas_cumsum_0_to_T) 
        Sigma2 = 2 * lamb * (sinh_thetas_cumsum_0_to_t) * (sinh_thetas_cumsum_t_to_T) / (sinh_thetas_cumsum_0_to_T) 
        Sigma = torch.sqrt(Sigma2) 

        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('thetas', thetas)
        register_buffer('thetas_cumsum_0_to_t', thetas_cumsum_0_to_t)
        register_buffer('thetas_cumsum_0_to_T', thetas_cumsum_0_to_T)
        register_buffer('thetas_cumsum_t_to_T', thetas_cumsum_t_to_T)
        register_buffer('sinh_thetas_cumsum_0_to_t', sinh_thetas_cumsum_0_to_t)
        register_buffer('sinh_thetas_cumsum_0_to_T', sinh_thetas_cumsum_0_to_T)
        register_buffer('sinh_thetas_cumsum_t_to_T', sinh_thetas_cumsum_t_to_T)
        register_buffer('Theta', Theta)
        register_buffer('Sigma2',Sigma2)
        register_buffer('Sigma', Sigma) 
        

    def predict_x_start_from_noise(self, x_t, t, mu, noise):
        return (
            ((x_t - mu - (extract(self.Sigma, t, x_t.shape) * noise)) / (extract(self.Theta, t, x_t.shape))) + mu
        )

    def predict_noise_from_x_start(self, x_t, t, mu, x_start):
        return (
            (x_t - mu - extract(self.Theta, t, x_t.shape)* (x_start - mu) ) / extract(self.Sigma, t, x_t.shape)
        )

    def model_predictions(self, x_t, mu,  t, clip_denoised=True): 
        model_output = self.model(x_t, mu,t)

        maybe_clip = partial(torch.clamp, min=-1.,max=1.) if clip_denoised else identity

        if self.objective == "pred_noise":
            noise = model_output
            x_start = self.predict_x_start_from_noise(x_t, t, mu, noise) 
            x_start = maybe_clip(x_start)
        elif self.objective == "pred_x_start":
            x_start = model_output
            x_start = maybe_clip(x_start)
            noise = self.predict_noise_from_x_start(x_t, t, mu, x_start)
        else:
            exit('please speficy the prediction mode')

        return ModelResPrediction(noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, x_input, gt, shape, last=True): 
        mu = x_input[0] 
        batch, device, total_timesteps, sampling_timesteps, objective = shape[
            0], self.thetas.device, self.num_timesteps, self.sampling_timesteps, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
   
        if self.condition:
            img = mu
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        
        if not last:
            img_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step',disable = True):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            preds = self.model_predictions(img, mu, time_cond)

            noise = preds.pred_noise 
            x_start = preds.pred_x_start 

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            Theta_now = self.Theta[time]
            Theta_next = self.Theta[time_next]
            Sigma_now = self.Sigma[time]
            Sigma_next = self.Sigma[time_next]
             
            if self.sampling_type == "pred_noise":
                if time == (self.num_timesteps-1):
                    img = mu
                else:
                    img = mu +  (Theta_next / Theta_now) * (img - mu) - (((Theta_next / Theta_now) * Sigma_now) - Sigma_next) * noise
            elif self.sampling_type == "pred_x_start":
                if time == (self.num_timesteps-1):
                    img = mu + Theta_next * (x_start - mu)   
                else:
                    img = mu +  (Sigma_next / Sigma_now) * (img - mu) + (Theta_next - (Theta_now * Sigma_next / Sigma_now)) *  (x_start - mu)  
            else:
                exit('Illegal objective')

            if not last:
                img_list.append(img)
    
        if self.condition:
            if not last:
                img_list = [mu]+img_list
            else:
                img_list = [mu, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)


    def sample(self, x_input=None, gt = None, batch_size=16, last=True):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddim_sample
        if self.condition:
            x_input = 2 * x_input - 1
            x_input = x_input.unsqueeze(0)

            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)

        samples = sample_fn(x_input, gt, size, last=last)
        return samples

    def q_sample(self, x_start, mu, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            mu + (x_start - mu) * extract(self.Theta, t, x_start.shape) + extract(self.Sigma, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self, loss_type='l1'):
        if loss_type == 'l1':
            return F.l1_loss
        elif loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def p_losses(self, imgs, t, noise=None):
        if isinstance(imgs, list):  
            x_start = 2 * imgs[0] - 1 
            mu = 2 * imgs[1] - 1      

        noise = default(noise, lambda: torch.randn_like(x_start)) # * (x_start - mu)

        b, c, h, w = x_start.shape

        x = self.q_sample(x_start, mu, t, noise=noise)
        model_out = self.model(x, mu,t)
        target = x_start

        loss = F.l1_loss(model_out, target)
        return loss

    def forward(self, img, *args, **kwargs):
        if isinstance(img, list):
            b, c, h, w, device, img_size, = * \
                img[0].shape, img[0].device, self.image_size
        else:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size

        t = torch.randint(0, int(self.num_timesteps), (b,), device=device).long()
        t = torch.clamp(t, min=0, max=self.num_timesteps-1)

        return self.p_losses(img, t, *args, **kwargs)

if __name__ == '__main__':
    print('Hello World')
