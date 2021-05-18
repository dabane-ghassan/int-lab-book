import numpy as np
import matplotlib.pyplot as plt
import os

from SLIP import imread
from SLIP import Image as Image_SLIP
import time

from PIL import Image

from skimage.color import rgb2hsv, rgb2lab, hsv2rgb, lab2rgb

import torch
torch.set_default_tensor_type('torch.DoubleTensor')

import imageio

from torch.nn.functional import interpolate 

mode= 'bilinear' #resizing : continuous transition, reduces edges,contrast
width = 32 #side of the cropped image used to build the pyramid
base_levels = 2 #downsampling/upsampling factor

N_batch = 4 #number of images 
pattern = 'i05june05_static_street_boston_p1010808'

n_sublevel = 2 #filters dictionnary, number of sublevels
n_azimuth = 12 #retinal transform characteristics 
n_theta = 12
n_phase = 2

#img_orig = Image.open('../data/i05june05_static_street_boston_p1010808.jpeg')

#im_color_npy = np.asarray(img_orig)
#N_X, N_Y, _ = im_color_npy.shape #dimensions 
#ds= 1
#im=Image_SLIP({'N_X': N_X, 'N_Y': N_Y, 'do_mask': True})

def color_encode(img, color_mode, n_batch, width, height, n_color):
    img = img.permute(0,2,3,1)
    img = img.view(n_batch * width, height, n_color)
    img = img.numpy().clip(0,255).astype('uint8')
    if color_mode == 'lab':
        img = rgb2lab(img)
    elif color_mode == 'hsv':
        img = rgb2hsv(img)
    img = torch.Tensor(img)
    img = img.view(n_batch, width, height, n_color)
    img = img.permute(0,3,1,2)
    return img


def cropped_pyramid(img_tens, 
                    width=width, 
                    base_levels=base_levels, 
                    color=True, 
                    do_mask=False, 
                    verbose=False, 
                    squeeze=False, 
                    gauss=False, 
                    n_levels=None,
                    color_mode='rgb'):
    
    n_batch, _, N_X, N_Y = img_tens.shape 
    # tensor of the images  (dimension 4)
    if n_levels == None:
        n_levels = int(np.log(np.max((N_X, N_Y))/width)/np.log(base_levels)) + 1 
        #computing the number of iterations cf:downsampling
    
    if do_mask and color_mode=='rgb' and not gauss:
        bias = 128
    else:
        bias = 0
    
    img_down = img_tens.clone()
    if color :
        img_crop = torch.zeros((n_batch, n_levels, 3, width, width)) + bias        
    else :
        img_crop = torch.zeros((n_batch, n_levels, 1, width, width)) + bias 
        #creating the tensor to store the cropped images while pyramiding
        #img_down = img_down.unsqueeze(2) # add color dim
    level_size=[[N_X, N_Y]]    
    
    for i_level in range(n_levels-1): 
        #each iteration -> residual_image = image - downsampled_cloned_image_reshaped_to_the_right_size 
        img_residual = img_down.clone()
        img_down = interpolate(img_down, scale_factor=1/base_levels, mode=mode) 
        #downsampling
        if not gauss:
            img_sub = interpolate(img_down, size=img_residual.shape[-2:], mode=mode)  
            #upsizing in order to substract

        if verbose: 
            print('Tensor shape=', img_down.shape, ', shape=', img_residual.shape)
        n_batch, n_color, h_res, w_res = img_residual.shape 
        #at each iteration the residual image size is reduced of a factor 1/base_levels (img_down the image downsampled at the previous iteration)
        
        if h_res > width or w_res > width :
            i_min = max(h_res // 2 - width // 2, 0)
            diff_i_min = i_min - (h_res // 2 - width // 2)
            if h_res > width:
                i_max = i_min + width
                diff_i_max = width
            else:
                i_max = i_min + h_res
                diff_i_max = diff_i_min + h_res
                
            j_min = max(w_res // 2 - width // 2, 0)
            diff_j_min = j_min - (w_res // 2 - width // 2)
            if w_res > width:
                j_max = j_min + width
                diff_j_max = width
            else:
                j_max = j_min + w_res
                diff_j_max = diff_j_min + w_res
            w_i = diff_i_max - diff_i_min
            w_j = diff_j_max - diff_j_min
            
            img_base_crop = img_residual[:, :, i_min:i_max, j_min:j_max]
            if color_mode != 'rgb':
                img_base_crop = color_encode(img_base_crop, 
                                             color_mode, 
                                             n_batch, 
                                             w_i,
                                             w_j,
                                             n_color)
            if not gauss:
                img_sub_crop = img_sub[:, :, i_min:i_max, j_min:j_max]
                if color_mode != 'rgb':
                    img_sub_crop = color_encode(img_sub_crop, 
                                             color_mode, 
                                             n_batch, 
                                             w_i,
                                             w_j,
                                             n_color)
                img_crop[:, 
                         i_level, 
                         :, 
                         diff_i_min:diff_i_max, 
                         diff_j_min:diff_j_max] = img_base_crop - img_sub_crop
            else:
                img_crop[:, 
                         i_level, 
                         :, 
                         diff_i_min:diff_i_max,
                         diff_j_min:diff_j_max] = img_base_crop
        else :
            i_min = width // 2 - h_res // 2
            i_max = i_min + h_res
            j_min = width // 2 - w_res // 2
            j_max = j_min + w_res
            img_base_crop = img_residual
            #print(img_base_crop.shape)
            if color_mode != 'rgb':
                img_base_crop = color_encode(img_base_crop, 
                                             color_mode, 
                                             n_batch, 
                                             h_res, 
                                             w_res,
                                             n_color)
            if not gauss:
                img_sub_crop = img_sub
                #print(img_sub_crop.shape)
                if color_mode != 'rgb':
                    img_sub_crop = color_encode(img_sub_crop, 
                                             color_mode, 
                                             n_batch, 
                                             h_res,
                                             w_res,
                                             n_color)

                img_crop[:, 
                         i_level, 
                         :, 
                         i_min:i_max, 
                         j_min:j_max] = img_base_crop - img_sub_crop  
            else:
                img_crop[:, 
                         i_level, 
                         :, 
                         i_min:i_max,                 
                         j_min:j_max] = img_base_crop                          
          
        level_size.append(list(img_down.shape[-2:]))
            
    n_batch, n_color, h_res, w_res = img_down.shape
    if h_res > width or w_res > width :
        i_min = max(h_res // 2 - width // 2, 0)
        diff_i_min = i_min - (h_res // 2 - width // 2)
        i_max = min(i_min + width, i_min + h_res)
        diff_i_max = i_max - i_min
        j_min = max(w_res // 2 - width // 2, 0)
        diff_j_min = j_min - (w_res // 2 - width // 2)
        j_max = min(j_min + width, j_min + w_res)
        diff_j_max = j_max - j_min
        img_crop[:, 
                 n_levels-1, 
                 :,
                 diff_i_min:diff_i_max, 
                 diff_j_min:diff_j_max] = img_down[:, :, i_min:i_max, j_min:j_max]
    else:
        i_min = width // 2 - h_res // 2
        i_max = i_min + h_res
        j_min = width // 2 - w_res // 2
        j_max = j_min + w_res
        img_crop[:, n_levels-1, :, 
                 i_min:i_max, 
                 j_min:j_max] = img_down 
    if color_mode != 'rgb':
        img_crop[:, n_levels-1, ...] = color_encode(img_crop[:, n_levels-1, ...], 
                                                     color_mode, 
                                                     n_batch, 
                                                     width,
                                                     width,
                                                     n_color)
    if not color :
        img_crop = img_crop.squeeze(2)
    if verbose: print('Top tensor shape=', img_down.shape, ', Final n_levels=', n_levels) 
        #print image's dimensions after downsampling, condition max(img_down.shape[-2:])<=width satisfied
    
    if do_mask :
        mask_crop = Image_SLIP({'N_X': width, 'N_Y': width, 'do_mask': True}).mask
        if color :
            for i in range(n_levels-1):
                if gauss:
                    img_crop[0,i,...] = (img_crop[0,i,...]-128)*mask_crop[:,:]+128
                else:
                    img_crop[0,i,...] *= mask_crop[:,:]
            img_crop[0,n_levels-1,...] = (img_crop[0,n_levels-1,...]-128)*mask_crop[:,:]+128 #*(1-mask_crop[:,:])
        else :
            print(img_crop.shape)
            img_crop *= mask_crop[np.newaxis,np.newaxis,:,:]    #+0.5*(1-mask_crop[np.newaxis,np.newaxis,:,:])            

    if squeeze:
        return img_crop.squeeze(), level_size
    else:
        return img_crop, level_size


def inverse_pyramid(img_crop, N_X=768, N_Y=1024, base_levels=base_levels, color=True, 
                    verbose=False, gauss=False, n_levels=None, color_test = False):
    N_batch = img_crop.shape[0]
    width = img_crop.shape[3]
    if n_levels == None:
        n_levels = int(np.log(np.max((N_X, N_Y))/width)/np.log(base_levels)) + 1 #number of cropped images = levels of the pyramid

    if color :
        img_rec = img_crop[:, -1, :, :, :]#.unsqueeze(1)
        h_res, w_res = img_rec.shape[-2:]
        for i_level in range(n_levels-1)[::-1]: # from the top to the bottom of the pyramid
            img_rec = interpolate(img_rec, scale_factor=base_levels, mode=mode) #upsampling (factor=base_levels)
            h_res, w_res = img_rec.shape[-2:]
            if gauss:
                if not color_test:
                    img_rec[:, :,
                    (h_res//2-width//2):(h_res//2+width//2),
                    (w_res//2-width//2):(w_res//2+width//2)] = img_crop[:, i_level, :, :, :]
                else:
                    img_rec[:, :,
                    (h_res//2-width//2):(h_res//2+width//2),
                    (w_res//2-width//2):(w_res//2+width//2)] = torch.max(img_rec[:, :,
                    (h_res//2-width//2):(h_res//2+width//2),
                    (w_res//2-width//2):(w_res//2+width//2)], img_crop[:, i_level, :, :, :])
                        
            else:
                img_rec[:, :,
                        (h_res//2-width//2):(h_res//2+width//2),
                        (w_res//2-width//2):(w_res//2+width//2)] += img_crop[:, i_level, :, :, :] #adding previous central crop to img_crop
        img_rec = img_rec[:, :, (h_res//2-N_X//2):(h_res//2+N_X//2), (w_res//2-N_Y//2):(w_res//2+N_Y//2)]

    else :
        img_rec = img_crop[:, -1, :, :].unsqueeze(1)
        for i_level in range(n_levels-1)[::-1]: # from the top to the bottom of the pyramid
            img_rec = interpolate(img_rec, scale_factor=base_levels, mode=mode) #upsampling (factor=base_levels)
            h_res, w_res = img_rec.shape[-2:]
            if gauss :
                img_rec[:, 0, (h_res//2-width//2):(h_res//2+width//2), (w_res//2-width//2):(w_res//2+width//2)] = img_crop[:, i_level, :, :] #adding previous central crop to img_crop
            else:
                img_rec[:, 0, (h_res//2-width//2):(h_res//2+width//2), (w_res//2-width//2):(w_res//2+width//2)] += img_crop[:, i_level, :, :] #adding previous central crop to img_crop
        img_rec = img_rec[:, :, (h_res//2-N_X//2):(h_res//2+N_X//2), (w_res//2-N_Y//2):(w_res//2+N_Y//2)]

    return img_rec


def saccade_to(img_color, orig, loc_data_ij):
    if type(img_color) == np.ndarray:
        img_copy = np.copy(img_color)
        img_copy=np.roll(img_copy, orig[0] - loc_data_ij[0], axis=0)
        img_copy=np.roll(img_copy, orig[1] - loc_data_ij[1], axis=1)
    elif type(img_color) == torch.Tensor:
        img_copy = torch.clone(img_color)
        img_copy = torch.roll(img_copy, (orig[0] - loc_data_ij[0],), (2,))
        img_copy = torch.roll(img_copy, (orig[1] - loc_data_ij[1],), (3,))
    return img_copy


def level_construct(img_crop_list, loc_data_ij, level_size, level, verbose=False):
    n_levels = int(np.log(np.max((N_X, N_Y))/width)/np.log(base_levels)) + 1
    orig = level_size[0]//2, level_size[1]//2
    img_lev = torch.zeros((1, 3, level_size[0], level_size[1]))
    img_div = torch.zeros((1, 3, level_size[0], level_size[1]))
    #print(img_lev.shape)
    nb_saccades= len(img_crop_list)
    for num_saccade in range(nb_saccades):
        sac_img =  img_crop_list[num_saccade][:, level, :, :, :]
        if level_size[0] < width:
            x_width = level_size[0]
            sac_img = sac_img[:,:,width//2 - level_size[0]//2:width//2 + level_size[0]//2,:]
        else:
            x_width = width
        if level_size[1] < width:
            y_width = level_size[1]
            sac_img = sac_img[:,:,:,width//2 - level_size[1]//2:width//2 + level_size[1]//2]
        else:
            y_width = width
        #print(sac_img.shape)

        loc = loc_data_ij[num_saccade] // 2**level
        img_lev = saccade_to(img_lev, orig, loc)
        img_lev[:,:,orig[0]-x_width//2:orig[0]+x_width//2, orig[1]-y_width//2:orig[1]+y_width//2] += sac_img
        img_lev = saccade_to(img_lev, loc, orig)
        img_div = saccade_to(img_div, orig, loc)
        img_div[:,:,orig[0]-x_width//2:orig[0]+x_width//2, orig[1]-y_width//2:orig[1]+y_width//2] += torch.ones_like(sac_img)
        img_div = saccade_to(img_div, loc, orig)
    # coefficients normalization
    indices_zero = (img_div == 0).nonzero().detach().numpy()
    img_div_npy = img_div.detach().numpy()
    for ind in indices_zero:
        img_div_npy[ind[0], ind[1], ind[2], ind[3]] = 1
    img_lev = img_lev // img_div_npy
    
    if level < n_levels-1:
        bias = 128
    else:
        bias = 0
    if verbose:
        plt.figure()
        img_aff = img_lev.detach().permute(0,2,3,1)[0,:,:,:].numpy()
        plt.imshow((img_aff+bias).astype('uint8'))
    return img_lev


def inverse_pyramid_saccades(img_crop_list, loc_data_ij, level_size, N_X=768, N_Y=1024, base_levels=base_levels, verbose=False):
    N_batch = img_crop_list[0].shape[0]
    width = img_crop_list[0].shape[3]
    n_levels = int(np.log(np.max((N_X, N_Y))/width)/np.log(base_levels)) + 1

    #img_rec = img_crop[:, -1, :, :, :] #.unsqueeze(1)
    img_rec = level_construct(img_crop_list, loc_data_ij, level_size[n_levels-1], level=n_levels-1, verbose=verbose)
    for i_level in range(n_levels-1)[::-1]: # from the top to the bottom of the pyramid
        img_rec = interpolate(img_rec, scale_factor=base_levels, mode=mode) #upsampling (factor=base_levels)
        h_res, w_res = img_rec.shape[-2:]
        img_lev = level_construct(img_crop_list, loc_data_ij, level_size[i_level], level=i_level, verbose=verbose)
        img_rec += img_lev #adding previous central crop to img_crop
    img_rec = img_rec[:, :, (h_res//2-N_X//2):(h_res//2+N_X//2), (w_res//2-N_Y//2):(w_res//2+N_Y//2)]

    return img_rec



from LogGabor import LogGabor
pe = {'N_X': width, 'N_Y': width, 'do_mask': False, 'base_levels':
          base_levels, 'n_theta': 24, 'B_sf': 0.6, 'B_theta': np.pi/12 ,
      'use_cache': True, 'figpath': 'results', 'edgefigpath':
          'results/edges', 'matpath': 'cache_dir', 'edgematpath':
          'cache_dir/edges', 'datapath': 'database/', 'ext': '.pdf', 'figsize':
          14.0, 'formats': ['pdf', 'png', 'jpg'], 'dpi': 450, 'verbose': 0}                 
#log-Gabor parameters
lg = LogGabor(pe)
print('Default lg shape=', lg.pe.N_X, lg.pe.N_Y)


def local_filter(azimuth, theta, phase, sf_0=.25, B_theta=lg.pe.B_theta, radius=width/4, lg=lg):

    x, y = lg.pe.N_X//2, lg.pe.N_Y//2         # center
    x += radius * np.cos(azimuth)
    y += radius * np.sin(azimuth)

    return lg.normalize(lg.invert(
        lg.loggabor(x, y, sf_0=sf_0, B_sf=lg.pe.B_sf, theta=theta, B_theta=B_theta) * np.exp(-1j * phase)))


def get_K(width=width, 
          n_sublevel = n_sublevel, 
          n_azimuth = n_azimuth, 
          n_theta = n_theta,
          n_phase = n_phase, 
          r_min = width/6, 
          r_max = width/3, 
          log_density_ratio = 2, 
          verbose=False,
          phase_shift=False,
          lg=lg): 
    
    #filter tensor K definition using Di Carlo's formulas
    
    K = np.zeros((width, width, n_sublevel, n_azimuth, n_theta, n_phase))
    for i_sublevel in range(n_sublevel):
        #sf_0 = .25*(np.sqrt(2)**i_sublevel)
        #radius = width/4/(np.sqrt(2)**i_sublevel)
        # Di Carlo / Retina Warp

        b = np.log(log_density_ratio)  / (r_max - r_min)
        a = (r_max - r_min) / (np.exp (b * (r_max - r_min)) - 1)
        c = r_min - a
        r_ref = r_min + i_sublevel * (r_max - r_min) / n_sublevel
        r_prim =  a * np.exp(b * (r_ref - r_min)) + c
        radius =  r_prim 
        d_r_prim = a * b * np.exp(b * (r_ref - r_min))
        p_ref = 4 * width / 32
        p_loc = p_ref * d_r_prim
        sf_0 = 1 / p_loc
        if verbose: print('i_sublevel, sf_0, radius', i_sublevel, sf_0, radius)
        for i_azimuth in range(n_azimuth):
            for i_theta in range(n_theta):
                for i_phase in range(n_phase):
                    if phase_shift:
                        azimuth = (i_azimuth+((i_sublevel+i_phase)%2)/2)*2*np.pi/n_azimuth
                    else:
                        azimuth = (i_azimuth+(i_sublevel%2)/2)*2*np.pi/n_azimuth
                    K[..., 
                      i_sublevel, 
                      i_azimuth, 
                      i_theta, 
                      i_phase] = local_filter(azimuth=azimuth,
                                              theta=i_theta*np.pi/n_theta + azimuth,
                                              phase=i_phase*np.pi/n_phase, 
                                              sf_0=sf_0, 
                                              radius=radius,
                                              lg=lg)
    K = torch.Tensor(K)

    if verbose: print('K shape=', K.shape)
    if verbose: print('K min max=', K.min(), K.max())

    return K

def get_K_inv(K,width=width, n_sublevel = n_sublevel, n_azimuth = n_azimuth, n_theta = n_theta,
          n_phase = n_phase, verbose=False):
    print('Filter tensor shape=', K.shape) 
    K_ = K.reshape((width**2, n_sublevel*n_azimuth*n_theta*n_phase))
    print('Reshaped filter tensor=', K_.shape)
    
    K_inv = torch.pinverse(K_) 
    print('Tensor shape=', K_inv.shape)
    K_inv =K_inv.reshape(n_sublevel, n_azimuth, n_theta, n_phase, width, width)

    return K_inv

def inverse_gabor(log_gabor_coeffs, K_inv, verbose=False):
    print('Tensor shape=', K_inv.shape)
    img_crop =  torch.tensordot(log_gabor_coeffs, K_inv,  dims=4)
    #img_crop+=128 # !! on residuals only !!
    return img_crop

def log_gabor_transform(img_crop, K, color=True):
    if color:
        chan_0 = torch.tensordot(img_crop[:,:,0,:,:], K,  dims=2).unsqueeze(2)
        chan_1 = torch.tensordot(img_crop[:,:,1,:,:], K,  dims=2).unsqueeze(2)
        chan_2 = torch.tensordot(img_crop[:,:,2,:,:], K,  dims=2).unsqueeze(2)
        return torch.cat((chan_0, chan_1, chan_2), dim=2)
    else:
        return torch.tensordot(img_crop[:,:,:,:], K,  dims=2)
    

