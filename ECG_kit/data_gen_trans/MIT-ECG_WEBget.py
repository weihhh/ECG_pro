import time,urllib
import socket,requests

socket.setdefaulttimeout(10)

fail_list=[]
def get_file(file_url,file_name):
    try:
        urllib.request.urlretrieve(file_url,file_name)
    except Exception as err:
        print('downloading error!:{0}---->{1}'.format(err,file_name))
        fail_list.append(file_name)
        return
    print('downloading success!----->{0}'.format(file_name))
    time.sleep(3)


ori_url='https://physionet.org/physiobank/database/mitdb/{}.{}'
for x in [range(104,125),range(200,234)]:   
    for i in x:
        for file_type in ['atr','dat','hea']:
            #name_list.append(ori_url.format(i,file_type))
            # print(ori_url.format(i,file_type))
            get_file(ori_url.format(i,file_type),'{}.{}'.format(i,file_type))          

for file in fail_list:
    print(file)
'''
110.atr
110.dat
110.hea
120.atr
120.dat
120.hea
204.atr
204.dat
204.hea
206.atr
206.dat
206.hea
211.atr
211.dat
211.hea
216.atr
216.dat
216.hea
218.atr
218.dat
218.hea
224.atr
224.dat
224.hea
225.atr
225.dat
225.hea
226.atr
226.dat
226.hea
227.atr
227.dat
227.hea
229.atr
229.dat
229.hea
'''