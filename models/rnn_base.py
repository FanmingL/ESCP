import torch
import copy
import os
import time

class RNNBase(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size_list, activation, layer_type, logger=None, aux_dim=0):
        super().__init__()
        assert len(activation) - 1 == len(hidden_size_list), "number of activation should be " \
                                                             "larger by 1 than size of hidden layers."
        assert len(activation) == len(layer_type), "number of layer type should equal to the activate"
        activation_dict = {
            'tanh': torch.nn.Tanh,
            'relu': torch.nn.ReLU,
            'sigmoid': torch.nn.Sigmoid,
            'leaky_relu': torch.nn.LeakyReLU,
            'linear': None
        }
        layer_dict = {
            'fc': torch.nn.Linear,
            # 'lstm': torch.nn.LSTM, # two output
            'gru': torch.nn.GRU    # one output
        }
        def decorate(module):
            return torch.jit.script(module)
        def fc_decorate(module):
            return module
        def rnn_decorate(module):
            return module
        lst_nh = input_size + aux_dim
        self.layer_list = []
        self.layer_type = copy.deepcopy(layer_type)
        self.activation_list = []
        self.rnn_hidden_state_input_size = []
        self.rnn_layer_type = []
        self.rnn_num = 0
        self.logger = logger
        for ind, item in enumerate(hidden_size_list):
            if self.layer_type[ind] == 'fc':
                self.layer_list.append(fc_decorate(layer_dict[self.layer_type[ind]](lst_nh, item)))
            else:
                self.rnn_num += 1
                self.layer_list.append(rnn_decorate(layer_dict[self.layer_type[ind]](lst_nh, item, batch_first=True)))
                self.rnn_hidden_state_input_size.append(item)
                self.rnn_layer_type.append(self.layer_type[ind])
            if activation_dict[activation[ind]] is not None:
                self.activation_list.append(activation_dict[activation[ind]]())
            else:
                self.activation_list.append(None)
            lst_nh = item + aux_dim
        if self.layer_type[-1] == 'fc':
            self.layer_list.append(fc_decorate(layer_dict[self.layer_type[-1]](lst_nh, output_size)))
        else:
            self.rnn_num += 1
            self.layer_list.append(rnn_decorate(layer_dict[self.layer_type[-1]](lst_nh, output_size, batch_first=True)))
            self.rnn_hidden_state_input_size.append(output_size)
            self.rnn_layer_type.append(self.layer_type[-1])
        if activation_dict[activation[-1]] is not None:
            self.activation_list.append(activation_dict[activation[-1]]())
        else:
            self.activation_list.append(None)
        # self.layer_list.append(torch.nn.Linear(lst_nh, output_size))
        # self.activation_list.append(activation_dict[activation[-1]]())
        self.total_module_list = self.layer_list + self.activation_list
        self._total_modules = torch.nn.ModuleList(self.total_module_list)
        self.input_size = input_size
        self.cumulative_forward_time = 0
        self.cumulative_meta_forward_time = 0
        assert len(self.layer_list) == len(self.activation_list), "number of layer should be equal to the number of activation"

    def make_init_state(self, batch_size, device=None):
        if device is None:
            device = torch.device("cpu")
        init_states = []
        for ind, item in enumerate(self.rnn_hidden_state_input_size):
            if self.rnn_layer_type[ind] == 'lstm':
                init_states.append((torch.zeros((1, batch_size, item), device=device),
                                torch.zeros((1, batch_size, item), device=device)))
            else:
                init_states.append(torch.zeros((1, batch_size, item), device=device))
        return init_states

    def meta_forward(self, x, hidden_state=None, require_full_hidden=False, aux_state=None):
        _meta_start_time = time.time()
        assert x.shape[-1] == self.input_size, f"inputting size does not match!!!! input is {x.shape[-1]}, expected: {self.input_size}"
        if hidden_state is None:
            hidden_state = self.make_init_state(x.shape[0], x.device)
        assert len(hidden_state) == self.rnn_num, f"rnn num does not match, input is {len(hidden_state)}, expected: {self.rnn_num}"
        x_dim = len(x.shape)
        assert x_dim >= 2, f"dim of input is {x_dim}, which < 1"
        if x_dim == 2:
            x = torch.unsqueeze(x, 0)
        aux_dim = -1
        if aux_state is not None:
            aux_dim = len(aux_state.shape)
        if aux_dim == 2:
            aux_state = torch.unsqueeze(aux_state, 0)
        rnn_count = 0
        output_hidden_state = []
        output_rnn = []
        for ind, layer in enumerate(self.layer_list):
            if aux_dim > 0:
                x = torch.cat((x, aux_state), -1)
            activation = self.activation_list[ind]
            layer_type = self.layer_type[ind]
            if layer_type == 'gru':
                _start_time = time.time()
                x, h = layer(x, hidden_state[rnn_count])
                _end_time = time.time()
                self.cumulative_forward_time += _end_time - _start_time
                rnn_count += 1
                output_hidden_state.append(h)
                if require_full_hidden:
                    output_rnn.append(x)
            else:
                _start_time = time.time()
                x = layer(x)
                _end_time = time.time()
                self.cumulative_forward_time += _end_time - _start_time
            if activation is not None:
                _start_time = time.time()
                x = activation(x)
                _end_time = time.time()
                self.cumulative_forward_time += _end_time - _start_time
        if x_dim == 2:
            x = torch.squeeze(x, 0)
        self.cumulative_meta_forward_time += time.time() - _meta_start_time
        if require_full_hidden:
            return x, output_hidden_state, output_rnn
        return x, output_hidden_state

    def copy_weight_from(self, src_net, tau):
        """I am target net, tau ~~ 1
            if tau = 0, self <--- src_net
            if tau = 1, self <--- self
        """
        with torch.no_grad():
            if tau == 0.0:
                self.load_state_dict(src_net.state_dict())
                return
            elif tau == 1.0:
                return
            for param, target_param in zip(src_net.parameters(True), self.parameters(True)):
                target_param.data.copy_(target_param.data * tau + (1-tau) * param.data)

    def info(self, info):
        if self.logger:
            self.logger.log(info)
        else:
            print(info)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.info(f'saving model to {path}..')
        torch.save(self.state_dict(), path)

    def load(self, path, **kwargs):
        self.info(f'loading from {path}..')
        map_location = None
        if 'map_location' in kwargs:
            map_location = kwargs['map_location']
        self.load_state_dict(torch.load(path, map_location=map_location))

    @staticmethod
    def append_hidden_state(hidden_state, data):
        for i in range(len(hidden_state)):
            if hidden_state[i] is None:
                hidden_state[i] = data[i]
            elif isinstance(hidden_state[i], tuple):
                hidden_state[i] = (torch.cat((hidden_state[i][0], data[i][0]), 1),
                                   torch.cat((hidden_state[i][1], data[i][1]), 1))
            else:
                hidden_state[i] = torch.cat((hidden_state[i], data[i]), 1)
        return hidden_state

    @staticmethod
    def pop_hidden_state(hidden_state):
        for i in range(len(hidden_state)):
            if hidden_state[i] is not None:
                if isinstance(hidden_state[i], tuple):
                    if hidden_state[i][0].shape[1] == 1:
                        hidden_state[i] = None
                    else:
                        hidden_state[i] = (hidden_state[i][0][:, 1:, :],
                                           hidden_state[i][1][:, 1:, :])
                else:
                    if hidden_state[i].shape[1] == 1:
                        hidden_state[i] = None
                    else:
                        hidden_state[i] = hidden_state[i][:, 1:, :]
        return hidden_state

    @staticmethod
    def get_hidden_length(hidden_state):
        if len(hidden_state) == 0:
            length = 0
        elif hidden_state[0] is not None:
            if isinstance(hidden_state[0], tuple):
                length = hidden_state[0][0].shape[1]
            else:
                length = hidden_state[0].shape[1]
        else:
            length = 0
        return length

if __name__ == '__main__':
    input_size = 32
    output_size = 4
    hidden_size = [64, 128, 32]
    activations = ["relu", "relu", "relu", "tanh"]
    layer_type = ["fc", "fc", "fc", "fc"]
    nn = RNNBase(input_size, output_size, hidden_size, activations, layer_type)
    init_state = nn.make_init_state(1, device=torch.device("cpu"))
    print(init_state, len(init_state))
    hidden_state = init_state
    for item in hidden_state:
        if isinstance(item, tuple):
            for item1 in item:
                print(item1.shape)
            print('\n')
        else:
            print(item.shape)
            print('\n')
    for i in range(10):
        print(i)
        out, hidden_state = nn.meta_forward(torch.randn((1, 1, input_size)), hidden_state)


