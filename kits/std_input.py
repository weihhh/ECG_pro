import sys

def read_stdin_nlines(nlines=1):
    '''
    从标准输入读取n行数据,默认为1
    '''
    input_list=[]
    while nlines:
        input_list.append(sys.stdin.readline().strip('\n'))#“字符串+\n”,需要用strip分割
        nlines-=1
    return input_list

print(read_stdin_nlines(3))

while 1:
    pass