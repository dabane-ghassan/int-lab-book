import os
import numpy as np
import matplotlib.pyplot as plt
#import SLIP for whitening and PIL for resizing
import SLIP
# copied from https://raw.githubusercontent.com/bicv/LogGabor/master/default_param.py
pe = {
    # Image
    # 'N_image' : None, #use all images in the folder
    'N_image' : 100, #use 100 images in the folder
    # 'N_image' : 10, #use 4 images in the folder
    'seed' : None, # a seed for the Random Number Generator (RNG) for picking images in databases, set to None xor set to a given number to freeze the RNG
    'N_X' : 256, # size of images
    'N_Y' : 256, # size of images
    # 'N_X' : 64, # size of images
    # 'N_Y' : 64, # size of images
    'noise' : 0.1, # level of noise when we use some
    'do_mask'  : True, # self.pe.do_mask
    'mask_exponent': 3., #sharpness of the mask
    # whitening parameters:
    'do_whitening'  : True, # = self.pe.do_whitening
    'white_name_database' : 'kodakdb',
    'white_n_learning' : 0,
    'white_N' : .07,
    'white_N_0' : .0, # olshausen = 0.
    'white_f_0' : .4, # olshausen = 0.2
    'white_alpha' : 1.4,
    'white_steepness' : 4.,
    'white_recompute' : False,
    # Log-Gabor
    #'base_levels' : 2.,
    'base_levels' : 1.618,
    'n_theta' : 24, # number of (unoriented) angles between 0. radians (included) and np.pi radians (excluded)
    'B_sf' : .4, # 1.5 in Geisler
    'B_theta' : 3.14159/18.,
    # PATHS
    'use_cache' : True,
    'figpath': 'results',
    'edgefigpath': 'results/edges',
    'matpath': 'cache_dir',
    'edgematpath': 'cache_dir/edges',
    'datapath': 'database',
    'ext' : '.pdf',
    'figsize': 14.,
    'formats': ['pdf', 'png', 'jpg'],
    'dpi': 450,
    'verbose': 0,
    }

class Display:
    def __init__(self, args, save=True):
        self.args = args
        # TODO: split dataloaders and give them the same minibatch size
        self.loader_train = get_data_loader(batch_size=args.minibatch_size, train=True, mean=args.mean, std=args.std,
                                            seed=args.seed)
        self.loader_test = get_data_loader(batch_size=args.test_batch_size, train=False, mean=args.mean, std=args.std,
                                           seed=args.seed)
        self.N_classes = len(self.loader_test.dataset.classes)

        np.random.seed(seed=args.seed + 1)
        # cache noise
        # path = f"/tmp/MotionClouds_{self.args.sf_0}_{self.args.B_sf}.npy"
        path = "/tmp/MotionClouds_%.3f_%.3f.npy" % (self.args.sf_0, self.args.B_sf)
        # print(path)
        if os.path.isfile(path):
            self.noise = np.load(path)
        else:
            self.noise = np.zeros((args.noise_batch_size, args.N_pic, args.N_pic))
            for i_noise in range(args.noise_batch_size):
                self.noise[i_noise, :, :], _ = MotionCloudNoise(sf_0=args.sf_0, B_sf=args.B_sf,
                                                                seed=self.args.seed + i_noise)
            if save:
                np.save(path, self.noise)

    def draw(self, data, i_offset=None, j_offset=None, radius=None, theta=None):
        # radial draw
        if radius is None:
            radius_f = np.abs(np.random.randn()) * self.args.offset_std
            radius = minmax(radius_f, self.args.offset_max)
            # print(radius_f, radius)
        if theta is None: theta = np.random.rand() * 2 * np.pi
        if i_offset is None: i_offset = int(radius * np.cos(theta))
        if j_offset is None: j_offset = int(radius * np.sin(theta))
        return self.place_object(data, i_offset, j_offset), i_offset, j_offset

    def place_object(self, data, i_offset, j_offset):
        if True:
            im_noise = self.noise[np.random.randint(self.args.noise_batch_size), :, :]
            im_noise = np.roll(im_noise, np.random.randint(self.args.N_pic), 0)
            im_noise = np.roll(im_noise, np.random.randint(self.args.N_pic), 1)
        else:
            im_noise = None
        return place_object(data, i_offset, j_offset, im_noise=im_noise, N_pic=self.args.N_pic,
                            contrast=self.args.contrast, noise=self.args.noise,
                            sf_0=self.args.sf_0, B_sf=self.args.B_sf, do_mask=self.args.do_mask)

    def show(self, ax, data_fullfield, ms=26, markeredgewidth=6, do_cross=True):
        ax.imshow(data_fullfield, cmap=plt.gray(), vmin=0, vmax=1)
        if do_cross: ax.plot([self.args.N_pic // 2], [self.args.N_pic // 2], '+', ms=ms,
                             markeredgewidth=markeredgewidth)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

def MotionCloudNoise(sf_0=0.125, B_sf=3., alpha=.0, N_pic=128, seed=42):
    import MotionClouds as mc
    mc.N_X, mc.N_Y, mc.N_frame = N_pic, N_pic, 1
    fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
    env = mc.envelope_gabor(fx, fy, ft, sf_0=sf_0, B_sf=B_sf, B_theta=np.inf, V_X=0., V_Y=0., B_V=0, alpha=alpha)

    z = mc.rectif(mc.random_cloud(env, seed=seed), contrast=1., method='Michelson')
    z = z.reshape((mc.N_X, mc.N_Y))
    return z, env


def get_data_loader(batch_size=100, train=True, mean=0.1307, std=0.3081, seed=2019):
    import torch
    torch.manual_seed(seed=seed)
    from torchvision import datasets, transforms
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    data_loader = torch.utils.data.DataLoader(
        # https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.MNIST
        datasets.MNIST('../data',
                       train=train,  # def the dataset as training data
                       download=True,  # download if dataset not present on disk
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((mean,), (std,))
                       ])),
        batch_size=batch_size,
        shuffle=True)
    return data_loader


def minmax(value, border):
    value = max(value, -border)
    value = min(value, border)
    return value


def do_offset(data, i_offset, j_offset, N_pic, data_min=None):
    # place data in a big image with some known offset
    N_stim = data.shape[0]
    center = (N_pic - N_stim) // 2
    if data_min is None:
        data_min = data.min()

    data_fullfield = data_min * np.ones((N_pic, N_pic))
    # print(data.shape, center+N_stim, i_offset, j_offset)
    data_fullfield[int(center + i_offset):int(center + N_stim + i_offset),
                   int(center + j_offset):int(center + N_stim + j_offset)] = data
    return data_fullfield


def place_object(data, i_offset, j_offset, im_noise=None, N_pic=128, contrast=1., noise=.5, sf_0=0.1, B_sf=0.1,
                 do_mask=True, do_max=False):
    # print(data.min(), data.max())
    data = (data - data.min()) / (data.max() - data.min())

    # place data in a big image with some known offset
    data_fullfield = do_offset(data=data, i_offset=i_offset, j_offset=j_offset, N_pic=N_pic, data_min=0.)

    # normalize data in [0, 1]
    data_fullfield = (data_fullfield - data_fullfield.min()) / (data_fullfield.max() - data_fullfield.min())
    # multiply by contrast
    data_fullfield *= contrast

    # add noise
    if noise > 0.:
        if im_noise is None:
            im_noise, _ = MotionCloudNoise(sf_0=sf_0, B_sf=B_sf)
        # print('im_noise in range=', im_noise.min(), im_noise.mean(), im_noise.max())
        im_noise = 2 * im_noise - 1  # go to [-1, 1] range
        im_noise = noise * im_noise
        # im_noise = .5 * im_noise + .5 # back to [0, 1] range
        # print('im_noise in range=', im_noise.min(), im_noise.mean(), im_noise.max())
        if do_max:
            data_fullfield[data_fullfield == 0] = -np.inf
            data_fullfield = np.max((im_noise, data_fullfield), axis=0)
        else:
            data_fullfield = np.sum((im_noise, data_fullfield), axis=0)
    # print(data_fullfield.min(), data_fullfield.max())

    # add a circular mask
    if do_mask:
        x, y = np.mgrid[-1:1:1j * N_pic, -1:1:1j * N_pic]
        R = np.sqrt(x ** 2 + y ** 2)
        mask = 1. * (R < 1)
        # print('mask', mask.min(), mask.max(), mask[0, 0])
        data_fullfield = data_fullfield * mask

    # normalize data in [0, 1]
    data_fullfield /= 2  # back to [0, 1] range
    data_fullfield += .5  # back to a .5 baseline
    data_fullfield = np.clip(data_fullfield, 0, 1)
    return data_fullfield




