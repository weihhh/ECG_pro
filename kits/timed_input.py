import sys,time,msvcrt

#调用了windows的库，msvrt

def timedInput(caption, default, timeout=5):
    start_time = time.time()
    sys.stdout.write('%s(%d秒自动跳过):' % (caption,timeout))#caption表示你需要显示的提示，timeout等待时间
    sys.stdout.flush()
    input = ''
    while True:
        ini=msvcrt.kbhit()#检测键盘输入的信号
        try:
            if ini:
                chr = msvcrt.getche()#从输入缓存中读出
                if ord(chr) == 13 :  # enter_key
                    break
                elif ord(chr) >= 32:
                    input += chr.decode()
        except Exception as e:
            pass
        if len(input) == 0 and time.time() - start_time > timeout:
            break
    print ('')  # needed to move to next line
    if len(input) > 0:
        return input
    else:
        return default
 
 
#使用方法
iscaiji=timedInput('是否继续？','y')
if iscaiji=='y' or iscaiji=='yes':
    print('输入---> ')
else:
    print('停止')