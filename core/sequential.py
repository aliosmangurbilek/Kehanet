from core.tensor import Tensor

class SimpleSequential:
    """
    Basit bir Sequential model sınıfı.
    """
    def __init__(self, layers):
        self._layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self._layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params
