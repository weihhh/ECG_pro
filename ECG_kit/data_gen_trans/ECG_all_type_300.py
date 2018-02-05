import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy import io
from scipy import stats
import pickle
from collections import Counter

#不仅5种，将全部心律不齐样本输入，不管个数，300一心拍


path=r'D:\aa_work\新学期项目及论文\心电图项目\数据集\MIT-BIH Arrhythmia Database\mat_data'

def get_files(path,recurse=False):
    '''
    得到指定目录下的所有文件列表，递归或者不递归
    '''
    files=[]
    if not recurse:
        for name in os.listdir(path) :
            fullname=os.path.join(path,name)
            if os.path.isfile(fullname):
                files.append(fullname)
        return files   
    else:
        for root ,dirs,names in os.walk(path) :
            for name in names:
                fullname=os.path.join(root,name)
                files.append(fullname)
        return files

def load_mat(folder,file_name):
    #loaddata
    folder=os.path.normcase(folder+'/')
    # folder='/Users/jig289/Dropbox/MATLAB/Projects/In_Progress/BMI/Processed_Data/' 
    data=io.loadmat(folder+file_name)
    return data



files_list=[i for i in get_files(path) if os.path.basename(i).rfind('ANNOTD.mat')!=-1]
# print(files_list[0],os.path.basename(files_list[0])[:3])


all_ecg_list=[]
all_ann_list=[]
for file in files_list:
    index=os.path.basename(file)[:3]
    print('处理文件序号： ',index) 
    
    #波形数据矩阵
    data_m=load_mat(path,'{}.mat'.format(index))
    ECG_matrxi=data_m['M']#array(648000,2),因matlab中存放的矩阵名叫M
    if index=='114':
        ii_sample=ECG_matrxi[:,1]#(648000,)
    else:
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
    
   # print 'annotation_time_matrix: ',annotation_time_matrix.shape

    ECG_spike=[]
    end_no=len(ii_sample)-1#尾sample序号
    
    #print 'end_no:',end_no
    
    right=len(annotation_time_matrix)
    left=0

    for i,R_time in enumerate(annotation_time_matrix.flatten()):
        R_samples=(R_time*360).astype(int)#float64,这里之前写成了R_time.astype(int)*360
        begin=R_samples-149
        # print(begin)
        end=R_samples+151
        # print(end)
        if begin>=0 and end<=end_no:
            ECG_spike.append(ii_sample[begin:end])#300个点
            #print R_samples
            #print 'begin',int(R_samples-149)
            #print 'end',end
            continue
        if begin<0:
            left=i+1
            continue
        if end>end_no:
            right=i
            continue

    
    plt.figure(2)

    ECG_spike=np.array(ECG_spike)#array of array
    #print 'ECG_spike_type',type(ECG_spike)
    #print index,'data: ',ECG_spike.shape
    #plt.plot(ECG_spike[0])
    #plt.show()
    all_ecg_list.append(ECG_spike)
    all_ann_list.append(annotation_matrix[left:right])
    #print index,'annotation: ',annotation_matrix[left:right].shape 


all_spike_matrix=np.concatenate(all_ecg_list,axis=0)
all_ann_matrix=np.concatenate(all_ann_list,axis=0)

    
new_data=all_spike_matrix
new_index=all_ann_matrix

#give matlab
# io.savemat('ECGdata.mat',{'ECGdata':new_data,'index':new_index})

print(new_data.shape)
print(new_index.shape)
all_counts=Counter(all_ann_matrix.flatten())
#print all_counts
#pickle
with open('all_type_data.pickle','wb') as f:
    pickle.dump([new_data,new_index],f)

print('finished downloading!')
#print 'all: ',all_spike_matrix.shape,all_ann_matrix.shape 
