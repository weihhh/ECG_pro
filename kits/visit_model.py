#访问者模式，严重依赖递归

class NodeVistor:
    def visit(self,node):
        methname='visit_'+type(node).__name__
        #通过传入类型名称确定处理函数，免去if来判断类型
        meth=getattr(self,methname,None)
        
        if meth is None:
            meth=self.generic_visit
        return meth(node)
    def generic_visit(self,node):
        raise RuntimeError('No {} method'.format('visit_'type(node).__name__))

class Evaluate(NodeVistor):
    #实际使用时继承后自己编写相关处理函数
    def visit_Number(self,node):
        return node.value