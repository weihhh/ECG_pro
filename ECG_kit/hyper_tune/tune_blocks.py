#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
用于各种模型调参
'''
#导入目标函数
import my_hyperopt
#导入调参包
from hyperopt import fmin,tpe,hp,partial
#设定参数空间
space = {"n_iter":hp.choice("n_iter",[i for i in range(30,50)]),
         "eta":hp.uniform("eta",0.05,0.5)}
#tpe.suggest 优化函数
algo = partial(tpe.suggest,n_startup_jobs=10)
#percept即最后需要调参的目标函数
best = fmin(my_hyperopt.percept,space,algo = algo,max_evals=50)