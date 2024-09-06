import rustlib

# val1: f64, val2: f64, weight1: f64, weight2: f64, bias:f64

class ScalerValue:
    def __init__(self, data, _children = (), _op = '' ) -> None:
        self.data = data
        self.grad = 0

        # Internal variables for constructing autograd graph
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op


    def __add__(self, other):
        other = other if isinstance(other, ScalerValue) else ScalerValue(other)  # Basically, operations between ScalerValue and other is not allowed.
        out = ScalerValue(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward

        return out


    def _radd__(self, other):
        return  other + self



    def __mul__(self, other):
        other = other if isinstance(other, ScalerValue) else ScalerValue(other)  # Basically, operations between ScalerValue and other is not allowed.
        out = ScalerValue(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad +=  other.data * out.grad   # Try doing this mathematically, you'lll find this intuitive.
            other.grad += self.data  * out.grad
        
        out._backward = _backward

        return out


    def _rmul__(self, other):
        return  other * self


    def __pow__(self, other):
        assert isinstance(other, (int, float)),  "Only dis"
        out = ScalerValue(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad +=  other * self.data**(other-1) * out.grad   # Try doing this mathematically. A little difficult.
        
        out._backward = _backward

        return out
    

    def relu(self):
        out = ScalerValue(0 if self.data < 0 else self.data, (self, ), 'ReLU')

        def _backward():
            self.grad +=  (out.data > 0) * out.grad
        
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []       # topological order of all the children in the graph.
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        # One variable at a time, apply the chain rule to get the gradient. 
        self.grad = 1
        for v in reversed(topo):
            v._backward()


    def __neg__(self):
        return self * -1
    

    def __sub__(self, other):
        return self + (-other)


    def __rsub__(self, other):
        return other + (-self)


    def __truediv__(self, other):
        return self * other**-1


    def __rtruediv__(self, other):
        return other * self**-1


    def __repr__(self):
        return f'ScalerValue(data={self.data}, grad={self.grad})' 

def py_eq_rst_comp(val1, val2, weight1, weight2, bias):
    x1 = ScalerValue(val1)
    x2 = ScalerValue(val2)

    w1 = ScalerValue(weight1)
    w2 = ScalerValue(weight2)

    b = ScalerValue(bias)

    x1w1 = (x1*w1)
    x2w2 = (x2*w2)

    x1w1x2w2 = x1w1 + x2w2 

    n = x1w1x2w2 + b 
    o = n.relu()

    print(f"O Data {o.data}")
    print(f"X1 Gradient {x1.grad}")

py_eq_rst_comp(2.0, 0.0, -3.0, 1.0, 6.88)
rustlib.scaler_backprop(2.0, 0.0, -3.0, 1.0, 6.88)

def test_scaler_backprop_rs(benchmark):
    result = benchmark(rustlib.scaler_backprop, 2.0, 0.0, -3.0, 1.0, 6.8813735870195432)

def test_scaler_backprop_py(benchmark):
    result = benchmark(py_eq_rst_comp, 2.0, 0.0, -3.0, 1.0, 6.8813735870195432)

# run the command `pytest main.py` to initiate benchmark tests.