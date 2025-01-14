class Value:

    def __init__(self,data,_op = '', _children =()):
        
        self.data = data
        self._backward = lambda : None
        self._prev = set(_children)
        self.grad = 0
        self._op = _op

    def __add__ (self,other):
        
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data,'+',(self,other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
            
        out._backward = _backward

        return out
    
    def backward(self):
        
        topo=[]
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
            topo.append(v)
        build_topo(self)

        self.grad = 1.0
        
        for v in reversed(topo):
            v._backward()
        
    def __mul__ (self,other):
        
        other = other if isinstance(other,Value) else Value(other)
        out = Value(other.data * self.data, '*', (self,other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

    def __pow__(self, other):
        assert isinstance(self,(int,float))
        out = (self.data ** other,f'**{other}',(self,))

        def _backward():
            self.grad += other*((self.data)**(other-1)) * out.grad
        
        out._backward = _backward
        

    def __neg__(self):
        return(self * -1)
    
    def __sub__(self, other):
        return(self + (-other))

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self,other):    
        return other + self
    
    def __rmul__(self,other):
        return self * other
    
    def __truediv__(self,other):
        return self * other**-1
    
    def __rtruediv__(self,other):
        return other * self**-1
    
    def __repr__(self):
        return f'Value(Data = {self.data}, Gradient = {self.grad})'
    