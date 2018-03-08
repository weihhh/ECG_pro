#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#pyw为后缀名则运行不带控制台

#心电数据程序
__author__='weizhang'

#图形界面包
from tkinter import *
from tkinter.filedialog import askdirectory
import tkinter.messagebox as messagebox
from tkinter import ttk 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  

import matplotlib.pyplot as plt      
from matplotlib.figure import Figure  

#xml 模块
from xml.dom.minidom import parse
import xml.dom.minidom

import os,pickle
import logging

#R波识别模块
import R_reco

#设置等级为WARNING,相当于不输出信息，程序中用的大多是info
logging.basicConfig(level=logging.WARNING ,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='ECG_GUI.log',
                filemode='w')

# ECG_leads_list=['MDC_ECG_LEAD_V1','MDC_ECG_LEAD_V2','MDC_ECG_LEAD_V3','MDC_ECG_LEAD_V4','MDC_ECG_LEAD_V5','MDC_ECG_LEAD_V6','MDC_ECG_LEAD_I','MDC_ECG_LEAD_II','MDC_ECG_LEAD_III','MDC_ECG_LEAD_AVF','MDC_ECG_LEAD_AVL','MDC_ECG_LEAD_AVR']
ECG_leads_list=['MDC_ECG_LEAD_I','MDC_ECG_LEAD_II','MDC_ECG_LEAD_III','MDC_ECG_LEAD_AVR','MDC_ECG_LEAD_AVL','MDC_ECG_LEAD_AVF','MDC_ECG_LEAD_V1','MDC_ECG_LEAD_V2','MDC_ECG_LEAD_V3','MDC_ECG_LEAD_V4','MDC_ECG_LEAD_V5','MDC_ECG_LEAD_V6']

Current_path=os.getcwd()
#存放xml文件的目录
ECG_XML_path=None
print('现在的工作目录、心电xml目录：{}、{}'.format(Current_path,ECG_XML_path))

#辅助函数
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
        self.fig=Figure()
        # self.fig.clf()
        self.canvas=FigureCanvasTkAgg(self.fig,master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack()    

        #当前选中文章,要去除之前加上的1--- 前缀
        self.filename=filename[filename.index('---')+3:]
        #选中的显示导联
        self.current_lead=StringVar()

        # 弹窗界面
        self.setup_UI()
    def setup_UI(self):
        # 第一行（两列）
        row2 =Frame(self)
        row2.pack(fill="x")
        self.comboxlist=ttk.Combobox(row2,textvariable=self.current_lead)
        self.comboxlist.pack(side=LEFT)
        self.comboxlist["values"]=ECG_leads_list
        self.comboxlist.current(0)
        # comboxlist.bind("<<ComboboxSelected>>",go)
        
        row3 = Frame(self)
        row3.pack(fill="x")

        noteLabel1 = Label(row3,text='输入数据长度！').pack()

        self.start_rowInput = Entry(row3)
        self.start_rowInput.pack()
        Button(row3, text="draw", command=self.plot_ECG).pack(side=LEFT)

        row6 = Frame(self)
        row6.pack(fill="x")
        Button(row6, text="quit", command=self.quit).pack(side=LEFT)
    def plot_ECG(self):
        #输入显示的单段长度
        length=int(self.start_rowInput.get()) if self.start_rowInput.get() else 3000
        #导联数据
        plot_data=self.get_ECG(self.comboxlist.get())#get()返回字符串
        #这里的数据还是字符串形式,强制转换
        plot_data=[int(i) for i in plot_data]
        #切分长度
        data_len=len(plot_data)
        rows=data_len//length+1
        print('length: {}'.format(data_len))
        
        self.fig.clf()
        
        # ax1=self.fig.add_subplot(1,1,1)
        # # max_value=max(plot_data)
        # # min_value=min(plot_data)
        # # my_y_ticks =range(min_value, max_value,(max_value-min_value)//2)
        # # plt.yticks(my_y_ticks)
        # ax1.plot(plot_data[:1000])

        for i in range(1,rows):
            ax1=self.fig.add_subplot(rows,1,i)
            # ax1.set_title(self.comboxlist.get())
            if i!=rows:
                ax1.plot(plot_data[length*(i-1):length*i])
            else:
                ax1.plot(plot_data[length*(i-1):])
        #调整子图间距
        self.fig.tight_layout()
        self.canvas.show()

    def get_ECG(self,lead):
        #读取对应导联数据文件
        if not hasattr(self,'data_list'):
            print('cache data_list!')
            with open(os.path.join(Current_path,'ECG_DATA',self.filename),'rt') as f:
                self.data_list=f.readlines()#读取整个文件所有行，保存在一个列表(list)变量中，每行作为一个元素
        #根据导联名称得到下标
        x=ECG_leads_list.index(lead)
        return self.data_list[x].strip('\n').split(',')
    
    def quit(self):
        self.destroy() # 销毁窗口


#主窗口
class Application(Frame):
    def __init__(self,master=None):
        Frame.__init__(self,master)
        self.pack()

        #当前显示的ECG数据文件
        self.ECG_TXT_files=self.load_ECG_list()
        self.ECG_XML_path=StringVar()
        self.ECG_XML_path.set(ECG_XML_path)

        self.createWidgets()

    def createWidgets(self):
        #界面安排
        self.noteLabel1 = Label(self, text='All_ECG in:  {}{}'.format('...',os.path.join(Current_path,'ECG_DATA')[-20:]))
        self.noteLabel1.grid(row=0,sticky=W)

        self.noteLabel2=Label(self)
        self.noteLabel2['textvariable']=self.ECG_XML_path      
        self.noteLabel2.grid(row=1,sticky=W)#显示当前读取文件名
        
        self.button1=Button(self,text='ECG_XML_path',command=self.select_xml_path)
        self.button1.grid(row=2,column=0,sticky=W)

        self.button2=Button(self,text='Extract_xmls',command=self.extract_xmls)
        self.button2.grid(row=2,column=1,sticky=W)

        self.button3=Button(self,text='annotations',command=self.generate_annotations)
        self.button3.grid(row=2,column=2,sticky=W)
        
        self.lb = Listbox(self)
        self.lb.bind('<Double-Button-1>',self.file_config)
        self.lb.grid(row=3,sticky=W,ipadx=200,ipady=200)
        
        self.update_list()

    def update_list(self):
        self.ECG_TXT_files=self.load_ECG_list()
        if self.ECG_TXT_files:
            for number,i in enumerate(self.ECG_TXT_files):
                self.lb.insert(END,'{}---{}'.format(number,i))
    def file_config(self,event):
        inputDialog = MyDialog(self.lb.get(self.lb.curselection()))
 
    def generate_annotations(self):
        for file in self.ECG_TXT_files:
            logging.info('annotating :{}'.format(file))
            with open(os.path.join(Current_path,'ECG_DATA',file),'rt') as f:
                #注意这里读取的时候要去掉首尾换行符,这里选取II导联进行识别R峰
                data_list=f.readlines()[ECG_leads_list.index('MDC_ECG_LEAD_II')].strip('\n').split(',')#读取整个文件所有行，保存在一个列表(list)变量中，每行作为一个元素
                data_list=[int(i) for i in data_list]
            threshold,R=R_reco.R_R_THRESHOLD(data_list,500)
            ann_time=R_reco.R_reco_wlt(data_list,ini_threshold=threshold,R_R=R,freq=500)
            #写入标注文件
            with open(os.path.join(Current_path,'ECG_DATA',os.path.splitext(file)[0][:-5]+'_ann.txt'),'wt') as f:
                for time in ann_time:
                    f.write(str(time)+',')
                    f.write('1'+'\n')

    def select_xml_path(self):
        global ECG_XML_path
        ECG_XML_path=askdirectory()
        self.ECG_XML_path.set(ECG_XML_path)
        print('select_xml_path : ',ECG_XML_path)

    def extract_xmls(self):
        #读取xml文件，将数据写入TXT，放入ECG_DATA中
        if not ECG_XML_path:
            messagebox.showinfo('提示','ECG_XML_path not selected!') 
            return 
        xml_files=[name for name in get_files(ECG_XML_path) if os.path.splitext(name)[1]=='.xml']
        for file in xml_files:
            logging.info('extract: {}'.format(file))
            result_dict=self.extract_one_xml(file)           
            (filepath,tempfilename) = os.path.split(file)
            (filename,extension) = os.path.splitext(tempfilename)
            with open(os.path.join(Current_path,'ECG_DATA',filename+'_wave.txt'),'wt') as f:
                for lead in ECG_leads_list:
                    if lead != 'TIME_ABSOLUTE':
                        for i,data in enumerate(result_dict[lead]['wave']):
                            if i ==0:
                                f.write(data)
                            else:
                                f.write(',')
                                f.write(data)
                        f.write('\n')
        self.update_list()
        messagebox.showinfo('提示','generate txt data finish!')    

    def extract_one_xml(self,XML_file):
        #打开xml文档
        # 使用minidom解析器打开 XML 文档
        DOMTree = xml.dom.minidom.parse(XML_file)
        #获得文档对象
        collection = DOMTree.documentElement

        print('当前文件名：',XML_file)
        print('当前节点名： {}'.format(collection.localName))
        if collection.hasAttribute("type"):
           print ("记录来自  : %s" % collection.getAttribute("type"))

        #长度1
        sequenceSet = collection.getElementsByTagName("sequenceSet")
        if len(sequenceSet)!=1:
            print('sequenceSet len out!!: ',len(sequenceSet))
            exit() 
        wave_sequence_list=sequenceSet[0].getElementsByTagName('sequence')
        print('wave sequence_list len: ',len(wave_sequence_list))

        result_dict={}
        for wave_sequence in wave_sequence_list:
            for i_1 in wave_sequence.childNodes:
                if i_1.localName=='code':
                    now_lead=i_1.getAttribute('code')
                    # print('now_lead: ',now_lead)
                if i_1.localName=='value':
                    if now_lead=='TIME_ABSOLUTE':
                        #保存[head,increment value,increment unit],三元素字典
                        result_dict[now_lead]={'head':[x for x in i_1.childNodes if x.localName=='head'][0].getAttribute('value'),'increment_value':[x for x in i_1.childNodes if x.localName=='increment'][0].getAttribute('value'),'increment_unit':[x for x in i_1.childNodes if x.localName=='increment'][0].getAttribute('unit')}
                    else:
                        result_dict[now_lead]={'origin_value':[x for x in i_1.childNodes if x.localName=='origin'][0].getAttribute('value'),'origin_unit':[x for x in i_1.childNodes if x.localName=='origin'][0].getAttribute('unit'),'scale_value':[x for x in i_1.childNodes if x.localName=='scale'][0].getAttribute('value'),'scale_unit':[x for x in i_1.childNodes if x.localName=='scale'][0].getAttribute('unit'),'wave':[x for x in i_1.childNodes if x.localName=='digits'][0].childNodes[0].data.split(' ')[:-1]}
        return result_dict

    def load_ECG_list(self):
        ECG_files=None
        if not os.path.exists(os.path.join(Current_path,'ECG_DATA')):
            if not ECG_XML_path:
                messagebox.showinfo('提示','No ECG_DATA and ECG_XML_path not selected!')
            # XML_files=[os.path.basename(i) for i in get_files(ECG_XML_path)]
                os.mkdir('./ECG_DATA')
                logging.info('CREAT ECG_DATA in: {}'.format(Current_path))
            # #CREAT TXT FILE
            # with open(os.path.join(ECG_XML_path,'pdf_files.pickle'),'wb') as f:
            #     pickle.dump(file_list,f)
            # logging.info('CREAT TXT FILES！ in: {}'.format(os.path.join(Current_path,'ECG_DATA')))

        else:
            ECG_files=[os.path.basename(i) for i in get_files(os.path.join(Current_path,'ECG_DATA')) if os.path.split(i)[1].find('_wave')!=-1]
        return ECG_files

app = Application()
# 设置窗口标题:
app.master.title('ECG')
# 主消息循环:
app.mainloop()