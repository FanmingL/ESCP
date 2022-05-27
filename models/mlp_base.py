import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rnn_base import RNNBase

class MLPBase(RNNBase):
    def __init__(self, input_size, output_size, hidden_size_list, activation):
        super().__init__(input_size, output_size, hidden_size_list, activation, ['fc'] * len(activation))

    def meta_forward(self, x, h=None, require_full_hidden=False):
        return super(MLPBase, self).meta_forward(x, [], False)

if __name__ == '__main__':
    hidden_layers = [256, 128, 64]
    hidden_activates = ['leaky_relu'] * len(hidden_layers)
    hidden_activates.append('tanh')
    nn = MLPBase(64, 4, hidden_layers, hidden_activates)
    for _ in range(5):
        x = torch.randn((3, 64))
        y, _ = nn.meta_forward(x)
        print(y)
