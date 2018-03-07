import os 


#判断缓存文件存在
username='wei'

if not os.path.exists('./tmp'):
    print('creat tmp dir!')
    os.mkdir('./tmp')
if os.path.exists('./tmp/{}.txt'.format(username)):
    print('ok')
else:
    print('fail')

#分离文件名，目录，后缀
file_path = "D:/test/test.py"
(filepath,tempfilename) = os.path.split(file_path)
(filename,extension) = os.path.splitext(tempfilename)


with open(file,'rt') as f:
    #注意这里读取的时候要去掉首尾换行符
    data_list=f.readlines()[0].strip('\n').split(',')#读取整个文件所有行，保存在一个列表(list)变量中，每行作为一个元素
