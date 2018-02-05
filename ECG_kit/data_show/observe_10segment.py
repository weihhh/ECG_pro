import numpy as np      #科学计算包 
import matplotlib.pyplot as plt      #python画图包   
import pickle


#unpickle
with open(r'ECG_10segment_data.pickle','rb') as f:
    all_ecg_list,all_ann_list,all_file_list=pickle.load(f)#array of array,(*, 300),(*,1) *样本数


sum_ecg=[]
sum_ann=[]

for i,file_no in enumerate(all_file_list):
    print('now file {}'.format(file_no))
    for j,segment in enumerate(all_ecg_list[i]):
        # if all_ann_list[i][j]!=[0]*11:
        #     print(all_ann_list[i][j])
        #     print(len(all_ecg_list[i][j]))
        #     plt.plot(all_ecg_list[i][j])
        #     plt.show()
        print(all_ecg_list[i][j].dtype)
        exit()
        sum_ecg.append(all_ecg_list[i][j])
        sum_ann.append(all_ann_list[i][j])
#pickle
with open('ECG_10segment_data_sum.pickle','wb') as f:
    pickle.dump([sum_ecg,sum_ann],f)