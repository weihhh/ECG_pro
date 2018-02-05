import numpy as np

all_ecg_list=[[[1,2,3],[23,5,6],[6,87,89]],[[11,22,33],[233,35,63],[63,873,892]]]
all_ann_list=[[1,2,3,4,2],[12,3,42,23,2]]
all_spike_matrix=np.concatenate(all_ecg_list,axis=0)
all_ann_matrix=np.concatenate(all_ann_list,axis=0)
print(all_spike_matrix.shape,all_spike_matrix)
print(all_ann_matrix.shape,all_ann_matrix)