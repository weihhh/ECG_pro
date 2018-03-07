#去除重复，返回是一个生成器
def dedupe(items,key=None):
    seen=set()
    for item in items:
        val=item if key is None else key(item)
        if val not in seen:
            yield item
            seen.add(val)
#key是处理函数，根据具体格式给，可用lamdba，
#需要去除重复的序列并且保持顺序，元素需可哈希，因要放进set

def main():
    testlist=[8,2,3,3,3,3,4,5,1]
    resultlist=dedupe(testlist)
    for i in resultlist :
        print(i)
if __name__ == '__main__':
    main()