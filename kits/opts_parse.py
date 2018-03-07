import optparse

def parse_options():
    parser=optparse.OptionParser()
    parser.add_option('-r','--recurse',dest='recurse',help='递归[default:%default]')
    parser.add_option('-c','--count',dest='count',type='int',help='线程或者进程数量[default:%default]')
    parser.add_option('-d','--debug',dest='debug',type='string',help='debug[default:%default]')
    parser.set_defaults(recurse=False,count=7,debug=False)
    opts,args=parser.parse_args()
    word=args[0]
    del args[0]
    return opts,word,args