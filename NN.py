import random
from Engine import Value

class Module:
    """Parametrization."""
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0      # Needed to set to 0 so that forward passes don't alter/mess the gradient values
        
    def parameters(self):
        return []

class Neuron(Module):
    """Structure the Neurons."""
    def __init__(self, nin, smoothing = 1):
        """Smoothing = 0 if no smoothing, 1 if ReLU, 2 if Sigmoid."""
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # Weights
        self.b = Value(0.0)                                         # Biases
        self.smoothing = smoothing
    def __call__(self, x):
        act = sum(((wi * xi) for wi, xi in zip(self.w, x)), self.b) # Activation function
        if self.smoothing == 1:
            return act.relu()
        elif self.smoothing == 2:
            return act.sigmoid()
        return act
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        if self.smoothing == 1:
            return f'ReLU Neuron({len(self.w)})'
        elif self.smoothing == 2:
            return f'Sigmoid Neuron({len(self.w)})'
        return f'Linear Neuon({len(self.w)})'
    
class Layer(Module):
    """Structure the Neuron Layer."""
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """Structure the MLP."""
    def __init__(self, nin, nouts, smoothing = 1):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], smoothing = smoothing) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"