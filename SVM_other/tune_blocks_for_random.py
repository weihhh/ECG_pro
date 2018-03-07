#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
用于各种模型调参
'''
#导入目标函数
import random_tree
#导入调参包
from hyperopt import fmin,tpe,hp,partial
#设定参数空间
space = {"n_estimators":hp.choice("n_estimators",[i for i in range(5,100)]),}
#tpe.suggest 优化函数
algo = partial(tpe.suggest,n_startup_jobs=10)
#percept即最后需要调参的目标函数
best = fmin(random_tree.random_forest_fun,space,algo = algo,max_evals=50)
print('best opt: ',best)
with open('random_forest_fun report.txt','a') as f:
    f.write('best: {}'.format(best))
