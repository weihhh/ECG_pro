import os

def get_files(path,recurse=False,full=True):
    '''
    得到指定目录下的所有文件列表，递归或者不递归,全路径或者只是文件名
    '''
    files=[]
    if not recurse:
        for name in os.listdir(path) :
            fullname=os.path.join(path,name)
            if os.path.isfile(fullname):
                if full:
                    files.append(fullname)
                else:
                    files.append(name)
        return files   
    else:
        for root ,dirs,names in os.walk(path) :
            for name in names:
                fullname=os.path.join(root,name)
                if full:
                    files.append(fullname)
                else:
                    files.append(name)
        return files   
            
# x=get_files(r'D:\aa_self')
# print(x)
