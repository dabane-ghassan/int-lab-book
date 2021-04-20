import torch
import easydict

from what_where.main import init
#args = init(filename='debug')
#args = init(filename='../data/2020-02-02')
#args = init()
args = init(filename='../data/2020-07-01')

from what_where.display import Display
from what_where.retina import Retina
from what_where.where import Where, WhereNet
from what_where.what import WhatNet
where = Where(args)
filename_train = args.filename + '_train.pt'
where.train(filename_train)
