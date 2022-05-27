from models.rnn_base import RNNBase
import torch
from torch.distributions import Normal
import numpy as np
import os


class Value(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, up_hidden_size, up_activations, up_layer_type,
                 ep_hidden_size, ep_activation, ep_layer_type, ep_dim, use_gt_env_feature,
                 logger=None, freeze_ep=False, enhance_ep=False, stop_pg_for_ep=False):
        super(Value, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.use_gt_env_feature = use_gt_env_feature
        # aux dim: we add ep to every layer inputs.
        aux_dim = ep_dim if enhance_ep else 0
        self.up = RNNBase(obs_dim + ep_dim + act_dim, 1, up_hidden_size, up_activations, up_layer_type, logger, aux_dim)
        self.ep = RNNBase(obs_dim + act_dim, ep_dim, ep_hidden_size, ep_activation, ep_layer_type, logger)
        self.ep_rnn_count = self.ep.rnn_num
        self.up_rnn_count = self.up.rnn_num
        # ep first, up second
        self.module_list = torch.nn.ModuleList(self.up.total_module_list + self.ep.total_module_list)
        self.min_log_std = -7.0
        self.max_log_std = 2.0
        self.sample_hidden_state = None
        self.ep_tensor = None
        self.freeze_ep = freeze_ep
        self.enhance_ep = enhance_ep
        self.stop_pg_for_ep = stop_pg_for_ep

    def get_ep(self, x, h, require_full_output=False):
        if require_full_output:
            ep, h, full_hidden = self.ep.meta_forward(x, h, require_full_output)
            if self.freeze_ep:
                ep = ep.detach()
            self.ep_tensor = ep
            return ep, h, full_hidden
        ep, h = self.ep.meta_forward(x, h)
        if self.freeze_ep:
            ep = ep.detach()
        self.ep_tensor = ep
        return ep, h

    def ep_h(self, h):
        return h[:self.ep_rnn_count]

    def up_h(self, h):
        return h[self.ep_rnn_count:]

    def make_init_state(self, batch_size, device):
        ep_h = self.ep.make_init_state(batch_size, device)
        up_h = self.up.make_init_state(batch_size, device)
        h = ep_h + up_h
        return h

    def meta_forward(self, x, lst_a, a, h, require_full_output=False, ep_out=None):
        ep_h = h[:self.ep_rnn_count]
        up_h = h[self.ep_rnn_count:]
        if not require_full_output:
            if self.use_gt_env_feature:
                up, up_h_out = self.up.meta_forward(torch.cat((x, a), -1), up_h)
                ep_h_out = []
            else:
                if ep_out is None:
                    ep, ep_h_out = self.get_ep(torch.cat((x, lst_a), -1), ep_h)
                else:
                    ep = ep_out
                    ep_h_out = []
                if self.stop_pg_for_ep:
                    ep = ep.detach()
                aux_input = ep if self.enhance_ep else None
                up, up_h_out = self.up.meta_forward(torch.cat((x, ep, a), -1), up_h, aux_state=aux_input)
            h_out = ep_h_out + up_h_out
            return up, h_out
        else:
            if self.use_gt_env_feature:
                up, up_h_out, up_full_hidden = self.up.meta_forward(torch.cat((x, a), -1), up_h,
                                                                    require_full_output)
                ep_h_out, ep_full_hidden = [], []
            else:
                if ep_out is None:
                    ep, ep_h_out, ep_full_hidden = self.get_ep(torch.cat((x, lst_a), -1), ep_h, require_full_output)
                else:
                    ep, ep_h_out = ep_out, []
                if self.stop_pg_for_ep:
                    ep = ep.detach()
                aux_input = ep if self.enhance_ep else None
                up, up_h_out, up_full_hidden = self.up.meta_forward(torch.cat((x, ep, a), -1), up_h,
                                                                    require_full_output, aux_state=aux_input)
            h_out = ep_h_out + up_h_out
            full_hidden = ep_full_hidden + up_full_hidden
            return up, h_out, full_hidden

    def forward(self, x, lst_a, a, h, ep_out=None):
        value_out, h_out = self.meta_forward(x, lst_a, a, h, ep_out=ep_out)
        return value_out, h_out

    def save(self, path, index=0):
        self.up.save(os.path.join(path, f'value_universe_policy{index}.pt'))
        self.ep.save(os.path.join(path, f'value_environment_probe{index}.pt'))

    def load(self, path, index=0, **kwargs):
        self.up.load(os.path.join(path, f'value_universe_policy{index}.pt'), **kwargs)
        self.ep.load(os.path.join(path, f'value_environment_probe{index}.pt'), **kwargs)

    @staticmethod
    def make_config_from_param(parameter):
        return dict(
            up_hidden_size=parameter.value_hidden_size,
            up_activations=parameter.value_activations,
            up_layer_type=parameter.value_layer_type,
            ep_hidden_size=parameter.ep_hidden_size,
            ep_activation=parameter.ep_activations,
            ep_layer_type=parameter.ep_layer_type,
            ep_dim=parameter.ep_dim,
            use_gt_env_feature=parameter.use_true_parameter,
            enhance_ep=parameter.enhance_ep,
            stop_pg_for_ep=parameter.stop_pg_for_ep
        )

    def copy_weight_from(self, src, tau):
        """
        I am target net, tau ~~ 1
        if tau = 0, self <--- src_net
        if tau = 1, self <--- self
        """
        self.up.copy_weight_from(src.up, tau)
        self.ep.copy_weight_from(src.ep, tau)

    def generate_hidden_state_with_slice(self, sliced_state: torch.Tensor, sliced_lst_action: torch.Tensor, sliced_action: torch.Tensor):
        """
        :param sliced_state: 0-dim: mini-trajectory index, 1-dim: batch_size, 2-dim: time step, 3-dim: feature index
        :param sliced_lst_action:
        :param slice_num:
        :return:
        """
        with torch.no_grad():
            mini_traj_num = sliced_state.shape[0]
            batch_size = sliced_state.shape[1]
            device = sliced_state.device
            hidden_states = []
            hidden_state_now = self.make_init_state(batch_size, device)
            for i in range(mini_traj_num):
                hidden_states.append(hidden_state_now)
                _, hidden_state_now = self.meta_forward(sliced_state[i], sliced_lst_action[i],
                                                        sliced_action[i], hidden_state_now)
        return hidden_states


    def generate_hidden_state(self, state: torch.Tensor, lst_action: torch.Tensor, action: torch.Tensor, slice_num):
        """
        :param sliced_state: 0-dim: mini-trajectory index, 1-dim: batch_size, 2-dim: time step, 3-dim: feature index
        :param sliced_lst_action:
        :param slice_num:
        :return:
        """
        with torch.no_grad():
            batch_size = state.shape[0]
            device = state.device
            hidden_states = []
            hidden_state_now = self.make_init_state(batch_size, device)
            _, _, full_hidden = self.meta_forward(state, lst_action, action, hidden_state_now, require_full_output=True)
            for i in range(len(full_hidden)):
                full_hidden[i] = torch.cat((hidden_state_now[i].squeeze(0).unsqueeze(1), full_hidden[i]), dim=1)
            idx = [i * slice_num for i in range(state.shape[1] // slice_num)]
            hidden_states = [item[:, idx].transpose(0, 1) for item in full_hidden]
            hidden_states_res = []
            for item in hidden_states:
                it_shape = item.shape
                hidden_states_res.append(item.reshape((1, it_shape[0] * it_shape[1], it_shape[2])))
        return hidden_states_res