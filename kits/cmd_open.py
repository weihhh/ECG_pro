import os
Current_path=os.getcwd()
print(Current_path)
os.chdir(r"D:\Program Files (x86)\kingpdf\Kingsoft PDF\10.1.0.6615\office6")
# os.chdir(r"D:\Program Files (x86)\kingpdf\Kingsoft PDF\10.1.0.6131\office6")
# y=os.popen(r'start .\wpspdf.exe "D:\aa_work\新学期项目及论文\心电图项目\心电论文\Heartbeat classification using morphological and dynamic features of ECG sig.pdf"')
os.popen(r'start .\wpspdf.exe "'+r'D:\aa_work\新学期项目及论文\心电图项目\心电论文'+'\\'+'Heartbeat classification using morphological and dynamic features of ECG sig.pdf'+'"')
os.chdir(Current_path)
print(os.getcwd())


