#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
用于各种模型调参
'''
#导入目标函数
import svm_tuned
#导入调参包
from hyperopt import fmin,tpe,hp,partial
#设定参数空间
space = {"C":hp.uniform("C",0.1,10),
            "gamma":hp.uniform("gamma",0.001,1)}
#tpe.suggest 优化函数
algo = partial(tpe.suggest,n_startup_jobs=10)
#percept即最后需要调参的目标函数
best = fmin(svm_tuned.svm_fun,space,algo = algo,max_evals=50)
print('best opt: ',best)
with open('svm_fun_report.txt','a') as f:
    f.write('best: {}'.format(best))
