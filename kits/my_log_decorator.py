import logging,os,tempfile,functools
if __debug__:#调试模式，即通常模式，如果运行在最优模式，命令行中，选项-O，就不会有日志
    logger=logging.getLogger('Logger')
    logger.setLevel(logging.DEBUG)
    handler=logging.FileHandler(os.path.join(tempfile.gettempdir(),'logged.log'))
    print('Note!:creat logfile at'+tempfile.gettempdir())#tempfile.gettempdir获得当前临时文档存取地址
    logger.addHandler(handler)
    
    def logged(function):
        "这是一个可以记录其修饰的任何函数的名称和参数和结果的修饰函数,使用方法参照grepword_thread文件中"
        @functools.wraps(function)
        def wrapper(*args,**kwargs):
            log='called:'+function.__name__+'('
            log+=','.join(['{0!r}'.format(a)for a in args]+['{0!s}={1!r}'.format(k,v)for k,v in kwargs.items()])
            result=exception=None
            try:
                result=function(*args,**kwargs)
                return result
            except Exception as err:
                exception=err
            finally:
                log+=((") ->"+str(result))if exception is None
                else "){0}:{1}".format(type(exception),exception))
                logger.debug(log)
                if exception is not None:
                    raise exception
        return wrapper
else :
    def logged(function):
        return function
'''
@logged
def discount_price(price,percentage,make_in=False):
    result=price+percentage
    return result
    
f=discount_price(2,3)
print(f)
'''