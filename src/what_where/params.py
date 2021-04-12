import os
import torch
from main import init, MetaML
from where import Where as ML
from what import WhatNet
import numpy as np
import sys
opts = dict(filename='../data/2020-02-02', verbose=0, log_interval=0, do_compute=False  if len(sys.argv) > 1 else True)
#opts = dict(filename='debug', verbose=0, log_interval=0)
print('do_compute =', opts['do_compute'])

print(50*'-')
print(' parameter scan')
print(50*'-')

if opts['do_compute']:
    args = init(**opts)
    #args.filename = '../data/2019-03-27'
    filename_train = args.filename + '_train.pt'
    if not(os.path.isfile(filename_train + '_lock')):
        open(filename_train + '_lock', 'w').close()
        print(50*'-')
        print('Default parameters')
        print(50*'-')
        from where import Where as ML
        from what import WhatNet
        ml = ML(args)
        ml.train(path=filename_train)
        # ml.main(path=args.filename)
        try:
            os.remove(filename_train + '_lock')
        except:
            pass

if True:
    args = init(**opts)
    mml = MetaML(args)
    if torch.cuda.is_available():
        mml.scan('no_cuda', [True, False])

    args = init(**opts)
    mml = MetaML(args)
    mml.scan('bias_deconv', [True, False])

results = {}
    
def update_results(results, parameter, accuracies, ci=0.01):
    from scipy.stats import beta

    if not parameter in results.keys(): results[parameter] = dict(value=[], accuracy=[], p_low=[], p_sup=[])
    for value in accuracies.keys():
        results[parameter]['value'].append(value)
        results[parameter]['accuracy'].append(accuracies[value][:-1].mean()*100)
        try:
            a1, b1, loc1, scale1 = beta.fit(accuracies[value][:-1], floc=0, fscale=1)
            p_low, p_sup = beta.ppf([ci, 1-ci], a=a1, b=b1)
            #print(p_low, p_sup)
            results[parameter]['p_low'].append(p_low*100)
            results[parameter]['p_sup'].append(p_sup*100)
        except:
            results[parameter]['p_low'].append(accuracies[value][:-1].mean()*100)
            results[parameter]['p_sup'].append(accuracies[value][:-1].mean()*100)
        
    return results
            
bases = [np.sqrt(2), 2, 2*np.sqrt(2)]
bases = [2]
bases = [np.sqrt(2), 2]

for base in bases if not args.filename == '../data/debug' else [2]:
    print(50*'-')
    print(' base=', base)
    print(50*'-')

    print(50*'-')
    print(' parameter scan : data')
    print(50*'-')
    args = init(**opts)
    mml = MetaML(args, base=base)
    for parameter in ['sf_0', 'B_sf', 'offset_std' , 'noise', 'contrast']: #
        accuracies = mml.parameter_scan(parameter)
        results = update_results(results, parameter, accuracies)
        
    print(50*'-')
    print(' parameter scan : network')
    print(50*'-')
    args = init(**opts)
    mml = MetaML(args)
    for parameter in ['dim1',
                      'bn1_bn_momentum',
                      'dim2',
                      'bn2_bn_momentum',
                      'p_dropout']:
        accuracies = mml.parameter_scan(parameter)
        results = update_results(results, parameter, accuracies)

    print(' parameter scan : learning ')
    args = init(**opts)
    mml = MetaML(args, base=base, tag='SGD')
    print(50*'-')
    print('Using SGD')
    print(50*'-')
    for parameter in ['lr', 'momentum', 'minibatch_size', 'epochs']:
        accuracies = mml.parameter_scan(parameter)
        results = update_results(results, parameter + '_sgd', accuracies)
    print(50*'-')
    print('Using ADAM')
    print(50*'-')
    args = init(**opts)
    args.do_adam = True
    mml = MetaML(args, base=base, tag='adam')
    for parameter in ['lr', 'momentum', 'minibatch_size', 'epochs']:
        if not (base == 2 and parameter=='epochs'): # HACK
            accuracies = mml.parameter_scan(parameter)
            results = update_results(results, parameter + '_adam', accuracies)

    print(50*'-')
    print(' parameter scan : retina')
    print(50*'-')
    args = init(**opts)
    mml = MetaML(args)
    for parameter in ['N_theta',
                      'N_azimuth',
                      'N_eccentricity',
                      'rho']:
        accuracies = mml.parameter_scan(parameter)
        results = update_results(results, parameter, accuracies)
        
