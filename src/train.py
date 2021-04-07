import torch
import easydict

from main import init
#args = init(filename='debug')
#args = init(filename='../data/2020-02-02')
#args = init()
args = init(filename='../data/2020-07-01')

from display import Display
from retina import Retina
from where import Where, WhereNet
from what import WhatNet
where = Where(args)
filename_train = args.filename + '_train.pt'
where.train(filename_train)
