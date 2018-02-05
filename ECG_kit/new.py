import scipy.io as sio
import matplotlib.pyplot as plt
import pywt
import math
import numpy as np

matfn = u'C:/Users/qsy/Desktop/Rselect/ecgdata.mat'
data=sio.loadmat(matfn)

ecgdata=data['ecgdata']
numdata = ecgdata.tolist()[0]  #矩阵转换列表

A=pywt.cwt(numdata,3, 'mexh')
coefs = A[0].tolist()[0]

freq=360;
num=len(numdata);
max_sum=0;
max_mean=0;
init_maximal=[];
maximal=[];  #极大值
maxaddr=[];  #极大值时间点
summit=[];
endsec=20;    #ECG信号的前20秒（用来计算初始阈值和R-R间期）
#######################################################计算初始阈值
for sec in range(endsec):
    everysec_data=coefs[freq*sec:freq*(sec+1)];
    max_sec = max(everysec_data);
    max_sum = max_sum + max_sec;
max_mean = max_sum/endsec;
thersold = max_mean*5/9; #初始阈值
#print(thersold)
#######################################################计算R-R间期
scount=0;
for i in range(endsec*freq):
    if coefs[i] > thersold:
        summit.append(i);
        scount=scount+1;
for j in range(scount):
    if coefs[summit[j]] > coefs[summit[j]-1] and coefs[summit[j]] > coefs[summit[j]+1]:
        init_maximal.append(summit[j]);
sumR = 0;
R= 0;
for i in range(len(init_maximal)-1):
    sumR = sumR + init_maximal[i+1] - init_maximal[i];
R = (4/5)*sumR/(len(init_maximal)-1);
#print(R)
###############################################

A=thersold;    #初始阈值
B=R;  #R-R间期
space=0;     #不应期200ms，初始化0


ptr_max=0;
count = 0;  #为了计算一个R峰后，未检测出R峰的时间，补偿法的变量
halfA=A/2;
k=0;
for i in range(num):
    if space != 0:
        space = space - 1;
    if coefs[i]>A and space == 0 :
      if coefs[i]>coefs[i-1] and coefs[i]>coefs[i+1]:
        space=72;    #不应期200ms
        ptr_max=coefs[i];
        maximal.append(coefs[i]);
        maxaddr.append(i);
        A = 0.75*A + 0.25*(5/9)*coefs[i];   # 实现自适应阈值法
        count=0;

    count=count+1;
    ### 实现补偿法策略
    if(count>=1.5*B):
        halfA=A/2;
        re=math.floor(i-1.1*B);
        for j in range(math.floor(1.1*B)):
            if coefs[re+j]>halfA and space == 0:
                if coefs[re+j]>coefs[re+j-1] and coefs[re+j]>=coefs[re+j+1]:
                    space = 72;
                    maximal.append(coefs[re+j]);
                    maxaddr.append(re+j);
                    A = 0.75*A + 0.25*(5/9)*coefs[re+j];
                    k=k+1;      ##统计通过补偿法得到的R峰值点
                    count=0;
                    break;
                
            
print(maxaddr)
print(len(maxaddr))


