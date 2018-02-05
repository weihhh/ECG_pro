import numpy as np

test=np.array([1.7,2.6,3.0,4.0,5.0],dtype=np.float)

print(np.delete(test,1))
print(test.dtype)

def adjust_len(input,target_len):
    #input array
    while len(input)<target_len:
        now_len=len(input)
        x=np.random.randint(1,now_len-1)
        print(x)
        insert_value=(input[x-1]+input[x])/2
        input=np.insert(input,x,insert_value)
    while len(input)>target_len:
        now_len=len(input)
        x=np.random.randint(1,now_len-1)
        input=np.delete(input,x)
    return input
print(adjust_len(test,6))