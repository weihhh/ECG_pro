import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy import io
from scipy import stats
import pickle
from collections import Counter


path=r'D:\aa_work\新学期项目及论文\心电图项目\数据集\MIT-BIH Arrhythmia Database\mat_data'

def load_mat(folder,file_name):
    #loaddata
    folder=os.path.normcase(folder+'/')
    # folder='/Users/jig289/Dropbox/MATLAB/Projects/In_Progress/BMI/Processed_Data/' 
    data=io.loadmat(folder+file_name)
    return data

index=100
#print '处理文件序号： ',index 

#波形数据矩阵
data_m=load_mat(path,'{}.mat'.format(index))
ECG_matrxi=data_m['M']#array(648000,2),因matlab中存放的矩阵名叫M
ii_sample=ECG_matrxi[:,0]#(648000,)
#print type(ii_sample)

#print 'ii_sample: ',ii_sample.shape

#注释
annotation=load_mat(path,'{}ANNOTD.mat'.format(index))
annotation_matrix=annotation['ANNOTD']#(2266，1)

#print 'annotation_matrix: ',annotation_matrix.shape

annotation_counts=Counter(annotation_matrix[:,0])
#print '标注类别及个数： ',annotation_counts 


#注释矩阵对应时间
annotation_time=load_mat(path,'{}ATRTIMED.mat'.format(index))
annotation_time_matrix=annotation_time['ATRTIMED']#(2266，1)

plt.figure(1)
# plt.plot(ii_sample[:5000])
# for i,ann in enumerate(annotation_matrix.flatten()[:100]):
#     plt.text(annotation_time_matrix.flatten()[i]*360,0.2,str(ann))


end_no=len(ii_sample)-1#尾sample序号
plt.plot(ii_sample[:5000])
for i,R_time in enumerate(annotation_time_matrix.flatten()):
        R_samples=(R_time*360).astype(int)#float64
        print(R_samples)
        begin=R_samples-149
        print(begin)
        end=R_samples+151
        print(end)

        plt.text(R_samples,0.2,str(annotation_matrix.flatten()[i]))
plt.show()
        #print end
        # if begin>=0 and end<=end_no:
        #     plt.plot(ii_sample[begin:end])
        #     plt.text(annotation_time_matrix.flatten()[i]*360,0.2,str(annotation_matrix.flatten()[i]))
        #     plt.show()
