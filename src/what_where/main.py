"""Parameters setting

Serves for initializing/scanning system parameters:

  - **MNIST LeNet parameters**:
    - ``w``: MNIST snippet width
    - ``mean``: LeNet input mean
    - ``std``: LeNet input std

  - **WhereNet inputs**:
    - 2D input image
      - ``N_pic`` : image width
    - Background noise:
      - ``noise``
      - ``contrast``
      - ``sf_0``: median spatial frequency
      - ``B_sf``: bandwidth around central frequency
    - Target positioning:
      - ``offset_std``
      - ``offset_max``
    - Logpolar encoding:
      - ``N_theta``
      - ``N_azimuth``
      - ``N_eccentricity``
      - ``N_phase``
      - ``rho``

  - **WhereNet setup**:
    - Parameters:
      - ``bias_deconv``
      - ``p_dropout``
      - ``lr``: Learning rate
      - ``do_adam``
      - ``bn1_bn_momentum``
      - ``bn2_bn_momentum``
      - ``momentum``
    - Layers:
      - ``dim1``
      - ``dim2``

  - **Train/test**:
    - ``minibatch_size``
    - ``train_batch_size``
    - ``test_batch_size``
    - ``noise_batch_size``
    - ``epochs``
    - ``log_interval``: period with which we report results for the loss

  - **Computation**:
    - ``num_processes``
    - ``no_cuda``

  - **Others**:
    - ``verbose``
    - ``filename``
    - ``seed``
    - ``N_cv``
    - ``do_compute``

"""

import os
import numpy as np
import time
import easydict
from PIL import Image

from torchvision.datasets.mnist import MNIST as MNIST_dataset
class MNIST(MNIST_dataset):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform((img, index))

        if self.target_transform is not None:
            target = self.target_transform((target, index))

        return img, target

def init(filename=None, verbose=1, log_interval=100, do_compute=True):
    if filename is None:
        do_recompute = True
        import datetime
        filename = '../../data/' + datetime.datetime.now().date().isoformat()
        print('Using filename=', filename)
    else:
        do_recompute = False

    import json
    filename_json = filename + '_param.json'
    if os.path.isfile(filename_json) and not do_recompute:
        with open(filename_json, 'r') as fp:
            args = json.load(fp)
            args = easydict.EasyDict(args)

    else:
        args = easydict.EasyDict(
                                # MNIST
                                w=28,
                                minibatch_size=100, # batch size
                                train_batch_size=50000, # size of training set
                                test_batch_size=10000,  # size of testing set
                                noise_batch_size=1000,
                                mean=0.1307,
                                std=0.3081,
                                what_offset_std=15,
                                what_offset_max=25,
                                # display
                                N_pic = 128,
                                offset_std = 30, #
                                offset_max = 34, # 128//2 - 28//2 *1.41 = 64 - 14*1.4 = 64-20
                                noise=.75, #0 #
                                contrast=.7, #
                                sf_0=0.1,
                                B_sf=0.1,
                                do_mask=True,
                                # foveation
                                N_theta=6,
                                N_azimuth=24,
                                N_eccentricity=10,
                                N_phase=2,
                                rho=1.41,
                                # network
                                bias_deconv=True,
                                p_dropout=.0,
                                dim1=1000,
                                dim2=1000,
                                # training
                                lr=5e-3,  # Learning rate
                                do_adam=True,
                                bn1_bn_momentum=0.5,
                                bn2_bn_momentum=0.5,
                                momentum=0.3,
                                epochs=60,
                                # simulation
                                num_processes=1,
                                no_cuda=False,
                                log_interval=log_interval, # period with which we report results for the loss
                                verbose=verbose,
                                filename=filename,
                                seed=2019,
                                N_cv=10,
                                do_compute=do_compute,
                                save_model=True,
                                )
        if filename == 'debug':
            args.filename = '../data/debug'
            args.train_batch_size = 100
            args.lr = 1e-2
            #args.noise = .5
            #args.contrast = .9
            #args.p_dropout = 0.
            args.epochs = 8
            args.test_batch_size = 20
            args.minibatch_size = 22
            #args.offset_std = 8
            args.N_cv = 2

        elif not do_recompute: # save if we want to keep the parameters
            with open(filename_json, 'w') as fp:
                json.dump(args, fp)

    return args

class MetaML:
    #from what import WhatNet
    #

    def __init__(self, args, base=2, N_scan=7, tag=''):
        self.args = args
        self.seed = args.seed

        self.base = base
        self.N_scan = N_scan
        self.tag = tag
        self.scan_folder = '../data/_tmp_scanning'
        os.makedirs(self.scan_folder, exist_ok=True)

    def test(self, args, seed):
        from where import WhereNet as ML
        # makes a loop for the cross-validation of results
        Accuracy = []
        for i_cv in range(self.args.N_cv):
            ml = ML(args)
            ml.train(seed=seed + i_cv)
            Accuracy.append(ml.test())
        return np.array(Accuracy)

    def protocol(self, args, seed):
        t0 = time.time()
        Accuracy = self.test(args, seed)
        t1 = time.time() - t0
        Accuracy = np.hstack((Accuracy, [t1]))
        return Accuracy

    def scan(self, parameter, values, verbose=True):
        import os
        print('scanning over', parameter, '=', values)
        seed = self.seed
        Accuracy = {}
        for value in values:
            if isinstance(value, int):
                value_str = str(value)
            else:
                value_str = '%.3f' % value
            filename = parameter + '_' + self.tag + '_' + value_str.replace('.', '_') + '.npy'
            path = os.path.join(self.scan_folder, filename)
            print ('For parameter', parameter, '=', value_str, ', ', end=" ")
            if not os.path.isfile(path + '_lock'):
                if not(os.path.isfile(path)) :# and self.args.do_compute:
                    open(path + '_lock', 'w').close()
                    try:
                        args = easydict.EasyDict(self.args.copy())
                        args[parameter] = value
                        Accuracy[value] = self.protocol(args, seed)
                        np.save(path, Accuracy[value])
                        os.remove(path + '_lock')
                    except ImportError as e:
                        print('Failed with error', e)
                else:
                    try:
                        Accuracy[value] = np.load(path)
                    except Exception as e:
                        print('Failed with error', e)
                if verbose:
                    try:
                        print('Accuracy={:.1f}% +/- {:.1f}%'.format(Accuracy[value][:-1].mean()*100, Accuracy[value][:-1].std()*100),
                      ' in {:.1f} seconds'.format(Accuracy[value][-1]))
                    except Exception as e:
                        print('Failed with error', e)

            else:
                print(' currently locked with ', path + '_lock')
            seed += 1
        return Accuracy

    def parameter_scan(self, parameter, display=False):
        if parameter in ['bn1_bn_momentum', 'bn2_bn_momentum', 'p_dropout']:
            values = np.linspace(0, 1, self.N_scan, endpoint=True)
        else:
            values = self.args[parameter] * np.logspace(-1, 1, self.N_scan, base=self.base, endpoint=True)
        if isinstance(self.args[parameter], int):
            # print('integer detected') # DEBUG
            values =  [int(k) for k in values]

        accuracies = self.scan(parameter, values)
        # print('accuracies=', accuracies)
        if display:
            fig, ax = plt.subplots(figsize=(8, 5))
            # TODO
        return accuracies
