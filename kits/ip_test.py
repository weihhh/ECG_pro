import socket
localIP = socket.gethostbyname(socket.gethostname())#得到本地ip
print("local ip:%s "%localIP)
 
ipList = socket.gethostbyname_ex(socket.gethostname())
for i in ipList:
    if i != localIP:
       print("external IP:%s"%i)