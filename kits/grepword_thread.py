import threading,queue
from opts_parse import parse_options
from get_files import get_files
import my_log_decorator
"python grepword_thread.py  a D:\\Documents\\GitHub\\python3-practice"

def main():
    opts,word,args=parse_options()
    filelist=get_files(args[0],opts.recurse)
    work_queue=queue.Queue()
    for i in range(opts.count):
        number='{0}:'.format(i+1)if opts.debug else ""
        worker=Worker(work_queue,word,number)
        worker.daemon=True
        worker.start()
    for filename in filelist:
        work_queue.put(filename)
    work_queue.join()
    
class Worker(threading.Thread):
    def __init__(self,work_queue,word,number):
        super().__init__()
        self.work_queue=work_queue
        self.word=word
        self.number=number
    def run(self):
        while True:
            try:
                filename=self.work_queue.get()
                self.process(filename)
            finally:
                self.work_queue.task_done()
    @my_log_decorator.logged
    def process(self,filename):
        previous=''
        BLOCK_SIZE=8000
        try:
            with open(filename,'rb') as fh:
                while True:
                    current=fh.read(BLOCK_SIZE)
                    if not current:
                        break
                    current=current.decode('utf8','ignore')
                    if (self.word in current or self.word in previous[-len(self.word):]+current[:len(self.word)]):
                        print('{0}{1}'.format(self.number,filename))
                        break
                    if len(current)!=BLOCK_SIZE:
                        break
                    previous=current
        except EnvironmentError as err:
            print('{0}{1}'.format(self.number,err))
            
if __name__ =='__main__':
    main()
                
 
  