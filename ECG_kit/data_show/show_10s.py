import numpy as np      #科学计算包 
import matplotlib.pyplot as plt      #python画图包   
import pickle
from sklearn import preprocessing
from collections import Counter
import os
from scipy import io

#注意显示的pickle文件！！！！！！！！

path=r'D:\aa_work\新学期项目及论文\心电图项目\数据集\MIT-BIH Arrhythmia Database\mat_data'

def load_mat(folder,file_name):
    #loaddata
    folder=os.path.normcase(folder+'/')
    # folder='/Users/jig289/Dropbox/MATLAB/Projects/In_Progress/BMI/Processed_Data/' 
    data=io.loadmat(folder+file_name)
    return data

#unpickle
with open(r'ECG_10S_data.pickle','rb') as f:
    all_ecg_list,all_ann_list,all_file_list=pickle.load(f)#array of array,(*, 300),(*,1) *样本数


all_counts=Counter(np.array(all_ann_list).flatten())
print('类别概览： ',all_counts)


for i,file in enumerate(all_ecg_list):
    if all_file_list[i]!='105':
        continue

    print('now file: {}'.format(all_file_list[i]))
    #注释
    annotation=load_mat(path,'{}ANNOTD.mat'.format(all_file_list[i]))
    annotation_matrix=annotation['ANNOTD']#(2266，1)

    #注释矩阵对应时间
    annotation_time=load_mat(path,'{}ATRTIMED.mat'.format(all_file_list[i]))
    annotation_time_matrix=annotation_time['ATRTIMED']#(2266，1)
    
    for j,data in enumerate(file):    
        if all_ann_list[i][j]!=[0]*10:
            print(all_ann_list[i][j])
            plt.figure(j)
            for x,R_time in enumerate(annotation_time_matrix.flatten()):
                
                R_samples=(R_time*360).astype(int)#float64
                if R_samples>(j+1)*3600:
                    break
                for num in range(0,3600,360):
                    plt.text(num,0.2,'h|',color='red')
                if R_samples>j*3600 and R_samples<(j+1)*3600:
                    plt.text(R_samples%3600,0.2,str(annotation_matrix.flatten()[x]))
            plt.plot(data)
            plt.show()

