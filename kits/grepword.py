import sys


if len(sys.argv) <3:
    print('please input more than 3 str')
    sys.exit()
#待搜索字
word=sys.argv[1]
files=sys.argv[2:]

for filename in files:
    for lino,line in enumerate(open(filename,'r',encoding='utf-8'),start=1):
        if word in line:
            print('{0}:{1}:{2:.40}'.format(filename,lino,line.rstrip()))