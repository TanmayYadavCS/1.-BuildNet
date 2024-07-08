class Value:
    """Created by Tanmay Yadav on July 7, 2024
    > Store a singular scalar and its gradients. 
    > Allows forward pass and backprogation within the model.
    > Multiple smoothing functions."""
    def __init__(self, data, _children = (), _op = ()) -> None:
        self.data = data                    # value storage
        self.grad = 0                       # gradient wrt to the last term
        # internal variables used for autograd construction
        self.backward = lambda: None        # backprop technique
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

    
