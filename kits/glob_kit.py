import glob
 
# The glob module finds all the pathnames matching a 
# specified pattern according to the rules used by the Unix shell,
#获取的文件名可以直接用

# >>> glob.glob('./[0-9].*')
# ['./1.gif', './2.txt']
# >>> glob.glob('*.gif')
# ['1.gif', 'card.gif']
# >>> glob.glob('?.gif')
# ['1.gif']

# >>> import glob
# >>> glob.glob('*.gif')
# ['card.gif']
# >>> glob.glob('.c*')
# ['.card.gif']

#获取指定目录下的所有图片
x=glob.glob(r"D:\aa_work\ECG\ECG_DATA_ALL\MIT_ECG_DATA\*.txt")
print(x)
# with open(x[0]) as f:
#     result=f.readlines()
#     print(result)
#获取上级目录的所有.py文件，返回的也是相对路径
x=glob.glob(r'**/*.py')
print(x) #相对路径
with open(x[0]) as f:
    result=f.readlines()
    print(result)

#父目录中的.py文件,可遍历对象
f = glob.iglob(r'../*.py')
 
print(f) #<generator object iglob at 0x00B9FF80>
 
for py in f:
    print(py)