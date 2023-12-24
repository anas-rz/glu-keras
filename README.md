# Multibackend GLU and its variants

A Keras 3 port of Gated Linear Units and its variants. Ported from [Rishit Dagli's Implementation](https://github.com/Rishit-dagli/GLU/tree/main)

# Installation

```shell
pip install glu_keras --upgrade
```

# Usage

```python
import keras
from glu_keras import SwiGLU

model = keras.Sequential()
model.add(keras.layers.Dense(units=10))
model.add(SwiGLU(bias = False, dim=-1, name='swiglu'))
```
