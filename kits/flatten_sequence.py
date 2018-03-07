from collections import Iterable

#扁平化序列，遇到嵌套会先打开
def flatten(items,ignore_types=(str,bytes)):
    for x in items:
        if isinstance(x,Iterable) and not isinstance(x,ignore_types):
            yield from flatten(x)
        else:
            yield x

# items=[1,2,[3,[6,7],5],9]
items=[[1,2],[3,4],[6,7],[2,5],[9,10]]
for i in flatten(items):
    print(i)