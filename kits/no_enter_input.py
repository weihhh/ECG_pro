import sys,time,msvcrt

def no_enter_input(note):
    print(note)
    ch=msvcrt.getch()#返回的bytes
    if ch==b'y':#或ord(ch)=97
        print('ok')
    else:
        print('cancel')

no_enter_input('输入y确定，或n取消！')