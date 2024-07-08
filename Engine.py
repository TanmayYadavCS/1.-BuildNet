import math
class Value:
    """Created by Tanmay Yadav on July 7, 2024
    > Store a singular scalar and its gradients. 
    > Allows forward pass and backprogation within the model.
    > Multiple smoothing functions."""
    def __init__(self, data, _children = (), _op = ()) -> None:
        self.data = data                    # value storage
        self.grad = 0                       # gradient wrt to the last term
        # internal variables used for autograd construction
        self._backward = lambda: None        # backprop technique
        self._prev = set(_children)         # children nodes
        self._op = _op                      # operation that originated the node
    

    def __add__(self, other):
        """Returns addition of two Value type scalars."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        # backprop
        def _backward():
            self.grad  += out.grad
            other.grad += out.grad
        # call it
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        """Returns multiplication of two Value type scalars."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        # backprop
        def _backward():
            self.grad  += other.grad * out.grad
            other.grad += self.grad  * out.grad
        # call it
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        """Returns power operation of two Value type scalars."""
        assert isinstance(other, (int, float)), "Only int/float powers allowed. Please retry with int/float values for the power."
        out = Value(self.data ** other, (self, other), f'**{other}')

        # backprop
        def _backward():
            self.grad  += (((other) * self.grad) ** (other - 1)) * out.grad
        # call it
        out._backward = _backward
        return out
    

    def relu(self):
        """Returns value processed through the ReLU smoothing function."""
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        # backprop
        def _backward():
            self.grad += (out.data > 0) * out.grad
        # call it
        out._backward = _backward
        return out
    
    def sigmoid(self):
        """Returns value processed through the Sigmoid smoothing function."""
        out = Value(1 / (1 + math.exp(-self.data)), (self,), 'Sigmoid')

        # backprop
        def _backward():
            self.grad += self.sigmoid(self.data) * (1 - self.sigmoid(self.data)) * out.grad
        # call it
        out._backward = _backward
        return out
    
    def backward(self):
        """Initiates the backward propogation from the parent node."""

        # topological sort
        topo = []
        visited = set()
        
        def build_topological(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topological(child)
                topo.append(v)
        
        build_topological(self)

        # go one variable at a time and apply the chain rule to get the subsequent gradients.
        self.grad = 1
        # since we will start with the last parent node
        for v in reversed(topo):
            v._backward()
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data = {self.data}, grad = {self.grad})"
