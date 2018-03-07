import my_charts
import charts
import os
from scipy import io


path=r'D:\aa_learning\pypro\python3-practice\kaggle\tensorflow\data_three_UAV2'


def load_mat(folder,file_name):
    #loaddata
    folder=os.path.normcase(folder+'/')
    # folder='/Users/jig289/Dropbox/MATLAB/Projects/In_Progress/BMI/Processed_Data/' 
    data=io.loadmat(folder+file_name)
    return data

mdata=load_mat(path,'data1.mat')
data=mdata['record1']
datax,datay=data[:,0].flatten(),data[:,1].flatten()

mdata1=load_mat(path,'data2.mat')
data1=mdata1['record2']
datax1,datay1=data1[:,0].flatten(),data1[:,1].flatten()

mdata2=load_mat(path,'data3.mat')
data2=mdata2['record3']
datax2,datay2=data2[:,0].flatten(),data2[:,1].flatten()

datalist=[]
for i in range(1,4):
    mdata=load_mat(path,'data{}.mat'.format(i))
    data=mdata['record{}'.format(i)]
    datax,datay=data[:,0].flatten(),data[:,1].flatten()
    datalist.append((datax,datay))

# # print(datax[:4],datay[:4])
# def fast_data(datax,fast_n):  
#     new_data=[]
#     for i,data in enumerate(datax):
#         if i%fast_n==0:
#             new_data.append(data)
#     return new_data
# dot_num=40000
# datax=fast_data(datax[:dot_num],1)
# datay=fast_data(datay[:dot_num],1)

print('begin！')
one_charts=my_charts.Charts_2d(datax1,datay1,[-1,101],[-1,101],dots_list=datalist)
# one_charts=my_charts.Charts_2d(datax1,datay1,[-1,101],[-1,101],dots_list=[(datax,datay)])
one_charts.write_gif(path+'\\plot_3line_grid.gif',fps=200)
# one_charts.write_video_2d(path+'\\he.mp4',fps=20)

# one_charts=charts.Charts(datax[:1000],datay[:1000],datay[:1000],[0,120],[0,120])
# one_charts.draw_dynamic()