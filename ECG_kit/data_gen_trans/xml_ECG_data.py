#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#读取xml文件数据，取出ECG波形数据

from xml.dom.minidom import parse
import xml.dom.minidom

import numpy as np      #科学计算包 
import matplotlib.pyplot as plt      #python画图包  


xml_file=r'D:\aa_work\新学期项目及论文\心电图项目\心电数据12-1\export\201801051344_03759378_金祖行.xml'

#打开xml文档
# 使用minidom解析器打开 XML 文档
DOMTree = xml.dom.minidom.parse(xml_file)
#获得文档对象
collection = DOMTree.documentElement

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

# print(result_dict)
for lead in result_dict:
    print('lead: ',lead)
    if lead=='TIME_ABSOLUTE':
        print('time: ',result_dict[lead])
    else:
        # plt.title(lead)
        print(lead)
        print('origin : {1} {0},scale : {3} {2}'.format(result_dict[lead]['origin_unit'],result_dict[lead]['origin_value'],result_dict[lead]['scale_unit'],result_dict[lead]['scale_value']))
        print(len(result_dict[lead]['wave']))
        # print(type(result_dict[lead]['wave'][0]))
        print('\n')
        plt.plot(result_dict[lead]['wave'][:2000])
        plt.show()
