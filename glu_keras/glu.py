import keras
from keras import ops

class GLU(keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = ops.split(x, 2, axis=self.dim)
        gate = ops.sigmoid(gate)
        x = out * gate
        return x


class Bilinear(keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(Bilinear, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = ops.split(x, 2, axis=self.dim)
        x = out * gate
        return x


class ReGLU(keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(ReGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = ops.split(x, 2, axis=self.dim)
        gate = ops.relu(gate)
        x = out * gate
        return x


class GeGLU(keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(GeGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = ops.split(x, 2, axis=self.dim)
        gate = keras.activations.gelu(gate)
        x = out * gate
        return x


class SwiGLU(keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(SwiGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = ops.split(x, 2, axis=self.dim)
        gate = keras.activations.swish(gate)
        x = out * gate
        return x


class SeGLU(keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(SeGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = ops.split(x, 2, axis=self.dim)
        gate = keras.activations.selu(gate)
        x = out * gate
        return x