import math
import random


def with_label(value, label):
    value._label = label
    return value


class Value:
    def __init__(self, value, prev=(), op='', label=''):
        self._label = label
        self.data = value
        self._prev = set(prev)
        self._op = op
        self._grad = 0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other):
        other = Value.get_value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def backward():
            self._grad += 1. * out.grad
            other._grad += 1. * out.grad
        out._backward = backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = Value.get_value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def backward():
            self._grad += other.data * out.grad
            other._grad += self.data * out.grad
        out._backward = backward

        return out

    def __rmul__(self, other):
        return self * other

    def zero_grad(self):
        self._grad = 0

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self._grad += out.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float are supported"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self._grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def backward():
            self._grad += (1 - t**2) * out.grad
        out._backward = backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def _build(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    _build(child)
                topo.append(v)

        _build(self)

        self._grad = 1
        for node in reversed(topo):
            node._backward()

    @staticmethod
    def get_value(x):
        if isinstance(x, Value):
            return x
        return Value(x, label=f'scalar({x})')

    @property
    def label(self):
        return self._label or ''

    @property
    def prev(self):
        return self._prev

    @property
    def op(self):
        return self._op

    @property
    def grad(self):
        return self._grad


class Neuron:
    def __init__(self, inputs_len):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(inputs_len)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, inputs_len, outputs_len):
        self.neurons = [Neuron(inputs_len) for _ in range(outputs_len)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, inputs_len, outputs_lens):
        sz = [inputs_len] + outputs_lens
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(outputs_lens))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
