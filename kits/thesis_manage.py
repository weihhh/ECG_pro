#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#pyw为后缀名则运行不带控制台

#论文管理
__author__='weizhang'

from tkinter import *
import tkinter.messagebox as messagebox

import os,pickle

Current_path=os.getcwd()
thesis_path=r'D:\aa_work\新学期项目及论文\心电图项目\心电论文'
print('现在的工作目录、论文目录：{}、{}'.format(Current_path,thesis_path))

# def thesis_option(event):
#     os.chdir(r"D:\Program Files (x86)\kingpdf\Kingsoft PDF\10.1.0.6615\office6")
#     print(os.getcwd())
#     os.popen(r'start .\wpspdf.exe "'+r'D:\aa_work\新学期项目及论文\心电图项目\心电论文'+'\\'+lb.get(lb.curselection())+'"')

def get_files(path,recurse=False):
    '''
    得到指定目录下的所有文件列表，递归或者不递归
    '''
    files=[]
    if not recurse:
        for name in os.listdir(path) :
            fullname=os.path.join(path,name)
            if os.path.isfile(fullname):
                files.append(fullname)
        return files   
    else:
        for root ,dirs,names in os.walk(path) :
            for name in names:
                fullname=os.path.join(root,name)
                files.append(fullname)
        return files 

#弹窗
class MyDialog(Toplevel):
    def __init__(self,filename):
        super().__init__()
        self.title('enjoy')
        
        #当前选中文章
        self.filename=filename
        # 弹窗界面
        self.setup_UI()
    def setup_UI(self):
        # 第一行（两列）
        row1 =Frame(self)
        row1.pack(fill="x")
        Button(row1, text='阅读', width=8,command=self.open_pdf).pack(side=LEFT)
        
        row2 = Frame(self)
        row2.pack(fill="x")
        self.red= IntVar()
        Checkbutton(row2,text='重要',variable=self.red,fg = 'red').pack(side=LEFT)

        row3 = Frame(self)
        row3.pack(fill="x")
        self.green= IntVar()
        Checkbutton(row3,text='已读',variable=self.green,fg = 'green').pack(side=LEFT)
        
        row4 = Frame(self)
        row4.pack(fill="x")
        self.black= IntVar()
        Checkbutton(row4,text='标记为未读',variable=self.black,fg = 'black').pack(side=LEFT)

        row5 = Frame(self)
        row5.pack(fill="x")
        self.del_item=IntVar()
        Checkbutton(row5,text='删除此论文',variable=self.del_item,fg = 'red').pack(side=LEFT)
        
        row6 = Frame(self)
        row6.pack(fill="x")
        Button(row6, text="取消", command=self.cancel).pack(side=RIGHT)
        Button(row6, text="确定", command=self.ok).pack(side=RIGHT)
    
    def ok(self):
        checklist=[self.red.get(), self.green.get(),self.black.get()]
        colorlist=['red','green','black']
        color='black'
        for i in range(3):
            if checklist[i]:
                color=colorlist[i]

        self.userinfo = (int(self.filename[:self.filename.index('---')]),color,self.del_item.get()) # 设置数据
        self.destroy() # 销毁窗口
    def cancel(self):
        self.userinfo = None # 空！
        self.destroy()
    def open_pdf(self):
        filename_r=self.filename[self.filename.index('---')+3:]
        if filename_r.endswith('pdf'):
            os.chdir(r"D:\Program Files (x86)\kingpdf\Kingsoft PDF\10.1.0.6621\office6")
            os.popen(r'start .\wpspdf.exe "'+r'D:\aa_work\新学期项目及论文\心电图项目\心电论文'+'\\'+filename_r+'"')
        if filename_r.endswith('caj'):
            os.chdir(r"D:\Program Files (x86)\TTKN\CAJViewer 7.2")
            os.popen(r'start .\CAJVieweru.exe "'+r'D:\aa_work\新学期项目及论文\心电图项目\心电论文'+'\\'+filename_r+'"')
        os.chdir(Current_path)
#主窗口
class Application(Frame):
    def __init__(self,master=None):
        Frame.__init__(self,master)
        self.pack()
        self.pdf_files=self.loaddata()
        self.createWidgets()

    def createWidgets(self):
        self.noteLabel1 = Label(self, text='All_thesis in:  {}{}'.format('...',thesis_path[-20:]))
        self.noteLabel1.grid(row=0,sticky=W)
        self.lb = Listbox(self)
        self.lb.bind('<Double-Button-1>',self.thesis_config)
        self.lb.grid(row=1,sticky=W,ipadx=200,ipady=200)
        self.update_button=Button(self,text='更新列表',command=self.reload)
        self.update_button.grid(row=2)
        for number,i in enumerate(self.pdf_files):
            self.lb.insert(END,'{}---{}'.format(number,i[1]))
            self.lb.itemconfig(number,{'fg':i[2]})#设置第i项颜色
            
    def thesis_config(self,event):
        res = self.ask_info()
        #处理返回值
        if res is None: return
        #res-->(序号，颜色)
        else:
            if res[2]:
                #删除此项
                del_item=self.pdf_files.pop(res[0])
                #更新序号
                for no,file in enumerate(self.pdf_files):
                    file[0]=no
                print('删除此项！--',del_item[1])
            else:
                self.pdf_files[res[0]][2]=res[1]

            #保存更改
            with open(os.path.join(thesis_path,'pdf_files.pickle'),'wb') as f:
                pickle.dump(self.pdf_files,f)
            print('论文状态修改完成！')
            self.update_list()
    def update_list(self):
        self.lb.delete(0,END)
        for number,i in enumerate(self.pdf_files):
            self.lb.insert(END,'{}---{}'.format(number,i[1]))
            self.lb.itemconfig(number,{'fg':i[2]})#设置第i项颜色
        print('更新显示列表！')

    def ask_info(self):
        
        inputDialog = MyDialog(self.lb.get(self.lb.curselection()))
        self.wait_window(inputDialog)
        return inputDialog.userinfo
    def loaddata(self):
        if not os.path.exists(os.path.join(thesis_path,'pdf_files.pickle')):
            files=[os.path.basename(i) for i in get_files(thesis_path)]
            file_list=[[number,name,'black'] for number,name in enumerate(files)]
            #pickle
            with open(os.path.join(thesis_path,'pdf_files.pickle'),'wb') as f:
                pickle.dump(file_list,f)
            print('新创建数据库！')

        else:
            #unpickle
            with open(os.path.join(thesis_path,'pdf_files.pickle'),'rb') as f:
                file_list=pickle.load(f)
            print('读取数据库！')
            # for i in file_list:
            #     print(i[0],i[1],i[2])
        return file_list
    def reload(self):
        #现文件夹中文件列表
        now_files=[os.path.basename(i) for i in get_files(thesis_path)]
        for file in now_files:
            if file not in [item[1] for item in self.pdf_files]:
                self.pdf_files.append(([0,file,'black']))
                print('新增!:--',file)
        #已删除文献显示灰色
        for file in self.pdf_files:
            if file[1] not in now_files:
                file[2]='grey'   

        #更新序号
        for no,file in enumerate(self.pdf_files):
            file[0]=no
        #保存更改
        with open(os.path.join(thesis_path,'pdf_files.pickle'),'wb') as f:
            pickle.dump(self.pdf_files,f)
        self.update_list()

app = Application()
# 设置窗口标题:
app.master.title('thesis')
# 主消息循环:
app.mainloop()