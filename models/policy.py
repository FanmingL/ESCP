from models.rnn_base import RNNBase
import torch
from torch.distributions import Normal
import numpy as np
import os
import time

class Policy(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, up_hidden_size, up_activations, up_layer_type,
                 ep_hidden_size, ep_activation, ep_layer_type, ep_dim, use_gt_env_feature,
                 rnn_fix_length, use_rmdm, share_ep,
                 logger=None, freeze_ep=False, enhance_ep=False, stop_pg_for_ep=False,
                 bottle_neck=False, bottle_sigma=1e-4):
        super(Policy, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.use_gt_env_feature = use_gt_env_feature
        # stop the gradient from ep when inferring action.
        self.stop_pg_for_ep = stop_pg_for_ep
        self.enhance_ep = enhance_ep
        self.bottle_neck = bottle_neck
        self.bottle_sigma = bottle_sigma
        # aux dim: we add ep to every layer inputs.
        aux_dim = ep_dim if enhance_ep else 0
        self.ep_dim = ep_dim
        self.up = RNNBase(obs_dim + ep_dim, act_dim * 2, up_hidden_size, up_activations, up_layer_type, logger, aux_dim)
        self.ep = RNNBase(obs_dim + act_dim, ep_dim, ep_hidden_size, ep_activation, ep_layer_type, logger)
        self.ep_temp = RNNBase(obs_dim + act_dim, ep_dim, ep_hidden_size, ep_activation, ep_layer_type, logger)
        self.ep_rnn_count = self.ep.rnn_num
        self.up_rnn_count = self.up.rnn_num
        # ep first, up second
        self.module_list = torch.nn.ModuleList(self.up.total_module_list + self.ep.total_module_list
                                               + self.ep_temp.total_module_list)
        self.soft_plus = torch.nn.Softplus()
        self.min_log_std = -7.0
        self.max_log_std = 2.0
        self.sample_hidden_state = None
        self.rnn_fix_length = rnn_fix_length
        self.use_rmdm = use_rmdm
        self.ep_tensor = None
        self.share_ep = share_ep
        self.freeze_ep = freeze_ep
        self.allow_sample = True
        self.device = torch.device('cpu')

    def set_deterministic_ep(self, deterministic):
        self.allow_sample = not deterministic

    def to(self, device):
        if not device == self.device:
            self.device = device
            if self.sample_hidden_state is not None:
                for i in range(len(self.sample_hidden_state)):
                    if self.sample_hidden_state[i] is not None:
                        self.sample_hidden_state[i] = self.sample_hidden_state[i].to(self.device)
            super().to(device)

    def get_ep_temp(self, x, h, require_full_output=False):
        if require_full_output:
            ep, h, full_hidden = self.ep_temp.meta_forward(x, h, require_full_output)
            if self.freeze_ep:
                ep = ep.detach()
            self.ep_tensor = ep

            return ep, h, full_hidden
        ep, h = self.ep_temp.meta_forward(x, h, require_full_output)
        if self.freeze_ep:
            ep = ep.detach()
        self.ep_tensor = ep
        return ep, h

    def apply_temp_ep(self, tau):
        self.ep.copy_weight_from(self.ep_temp, tau)

    def get_ep(self, x, h, require_full_output=False):
        # self.ep_tensor = torch.zeros((x.shape[0], x.shape[1], self.ep_dim), device=x.device, dtype=x.dtype)
        # return self.ep_tensor, h
        if require_full_output:
            ep, h, full_hidden = self.ep.meta_forward(x, h, require_full_output)
            if self.share_ep or self.freeze_ep:
                ep = ep.detach()
            self.ep_tensor = ep
            # if self.use_rmdm:
            #     ep = ep.detach()
            # ep = ep / torch.clamp_min(ep.pow(2).mean(dim=-1, keepdim=True).sqrt(), 1e-5)
            return ep, h, full_hidden
        ep, h = self.ep.meta_forward(x, h, require_full_output)
        if self.share_ep or self.freeze_ep:
            ep = ep.detach()
        self.ep_tensor = ep
        # if self.use_rmdm:
        #     ep = ep.detach()
        # ep = ep / torch.clamp_min(ep.pow(2).mean(dim=-1, keepdim=True).sqrt(), 1e-5)
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

    def make_init_action(self, device=torch.device('cpu')):
        return torch.zeros((1, self.act_dim), device=device)
    
    def tmp_ep_res(self, x, lst_a, h):
        ep_h = h[:self.ep_rnn_count]
        ep, ep_h_out = self.get_ep_temp(torch.cat((x, lst_a), -1), ep_h)
        return ep

    def meta_forward(self, x, lst_a, h, require_full_output=False):
        ep_h = h[:self.ep_rnn_count]
        up_h = h[self.ep_rnn_count:]
        if not require_full_output:
            if not self.use_gt_env_feature:
                ep, ep_h_out = self.get_ep(torch.cat((x, lst_a), -1), ep_h)
                if self.bottle_neck and self.allow_sample:
                    ep = ep + torch.randn_like(ep) * self.bottle_sigma
                if self.stop_pg_for_ep:
                    ep = ep.detach()
                aux_input = ep if self.enhance_ep else None
                up, up_h_out = self.up.meta_forward(torch.cat((x, ep), -1), up_h, aux_state=aux_input)
            else:
                up, up_h_out = self.up.meta_forward(x, up_h)
                ep_h_out = []
        else:
            if not self.use_gt_env_feature:
                ep, ep_h_out, ep_full_hidden = self.get_ep(torch.cat((x, lst_a), -1), ep_h, require_full_output)
                if self.bottle_neck and self.allow_sample:
                    ep = ep + torch.randn_like(ep) * self.bottle_sigma
                if self.stop_pg_for_ep:
                    ep = ep.detach()
                aux_input = ep if self.enhance_ep else None
                up, up_h_out, up_full_hidden = self.up.meta_forward(torch.cat((x, ep), -1), up_h, require_full_output, aux_state=aux_input)
            else:
                up, up_h_out, up_full_hidden = self.up.meta_forward(x, up_h, require_full_output)
                ep_h_out = []
                ep_full_hidden = []
            h_out = ep_h_out + up_h_out
            return up, h_out, ep_full_hidden + up_full_hidden
        h_out = ep_h_out + up_h_out
        return up, h_out

    def forward(self, x, lst_a, h, require_log_std=False):
        policy_out, h_out = self.meta_forward(x, lst_a, h)
        mu, log_std = policy_out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()
        if require_log_std:
            return mu, std, log_std, h_out
        return mu, std, h_out

    def rsample(self, x, lst_a, h):
        mu, std, log_std, h_out = self.forward(x, lst_a, h, require_log_std=True)
        # sample = torch.randn_like(mu).detach() * std + mu
        noise = torch.randn_like(mu).detach() * std
        sample = noise + mu
        log_prob = (- 0.5 * (noise / std).pow(2) - (log_std + 0.5 * np.log(2 * np.pi))).sum(-1, keepdim=True)
        # log_prob = dist.log_prob(sample).sum(-1, keepdim=True)

        log_prob = log_prob - (2 * (- sample - self.soft_plus(-2 * sample) + np.log(2))).sum(-1, keepdim=True)
        return torch.tanh(mu), std, torch.tanh(sample), log_prob, h_out

    def save(self, path):
        self.up.save(os.path.join(path, 'universe_policy.pt'))
        self.ep.save(os.path.join(path, 'environment_probe.pt'))

    def load(self, path, **kwargs):
        self.up.load(os.path.join(path, 'universe_policy.pt'), **kwargs)
        self.ep.load(os.path.join(path, 'environment_probe.pt'), **kwargs)

    @staticmethod
    def make_config_from_param(parameter):
        return dict(
            up_hidden_size=parameter.up_hidden_size,
            up_activations=parameter.up_activations,
            up_layer_type=parameter.up_layer_type,
            ep_hidden_size=parameter.ep_hidden_size,
            ep_activation=parameter.ep_activations,
            ep_layer_type=parameter.ep_layer_type,
            ep_dim=parameter.ep_dim,
            use_gt_env_feature=parameter.use_true_parameter,
            rnn_fix_length=parameter.rnn_fix_length,
            use_rmdm=parameter.use_rmdm,
            share_ep=parameter.share_ep,
            enhance_ep=parameter.enhance_ep,
            stop_pg_for_ep=parameter.stop_pg_for_ep,
            bottle_neck=parameter.bottle_neck,
            bottle_sigma=parameter.bottle_sigma
        )

    def inference_init_hidden(self, batch_size, device=torch.device("cpu")):
        if self.rnn_fix_length is None or self.rnn_fix_length == 0:
            self.sample_hidden_state = self.make_init_state(batch_size, device)
        else:
            self.sample_hidden_state = [None] * len(self.make_init_state(batch_size, device))

    def inference_check_hidden(self, batch_size):
        if self.sample_hidden_state is None:
            return False
        if len(self.sample_hidden_state) == 0:
            return True
        if self.rnn_fix_length is not None and self.rnn_fix_length > 0:
            return True
        if isinstance(self.sample_hidden_state[0], tuple):
            return self.sample_hidden_state[0][0].shape[0] == batch_size
        else:
            return self.sample_hidden_state[0].shape[0] == batch_size

    def inference_rnn_fix_one_action(self, state, lst_action):
        if self.use_gt_env_feature:
            mu, std, act, logp, self.sample_hidden_state = self.rsample(state, lst_action, self.sample_hidden_state)
            return mu, std, act, logp, self.sample_hidden_state

        while RNNBase.get_hidden_length(self.sample_hidden_state) >= self.rnn_fix_length:
            self.sample_hidden_state = RNNBase.pop_hidden_state(self.sample_hidden_state)
        self.sample_hidden_state = RNNBase.append_hidden_state(self.sample_hidden_state,
                                                               self.make_init_state(1, state.device))
        # print('1: ', self.sample_hidden_state)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
            lst_action = lst_action.unsqueeze(0)
        length = RNNBase.get_hidden_length(self.sample_hidden_state)
        # length = max(length, 1)
        # state = state.repeat_interleave(length, dim=0)
        state = torch.cat([state] * length, dim=0)
        # print('input: ', state[0])
        # lst_action = lst_action.repeat_interleave(length, dim=0)
        lst_action = torch.cat([lst_action] * length, dim=0)
        mu, std, act, logp, self.sample_hidden_state = self.rsample(state, lst_action, self.sample_hidden_state)
        # print('2: ', self.sample_hidden_state)

        return mu, std, act, logp, self.sample_hidden_state

    def inference_one_step(self, state, deterministic=True):
        self.set_deterministic_ep(deterministic)
        with torch.no_grad():
            lst_action = state[..., :self.act_dim]
            state = state[..., self.act_dim:]
            if self.rnn_fix_length is None or self.rnn_fix_length == 0 or len(self.sample_hidden_state) == 0:
                mu, std, act, logp, self.sample_hidden_state = self.rsample(state, lst_action, self.sample_hidden_state)
            else:
                while RNNBase.get_hidden_length(self.sample_hidden_state) < self.rnn_fix_length - 1 and not self.use_gt_env_feature:
                    _, _, _, _, self.sample_hidden_state = self.inference_rnn_fix_one_action(torch.zeros_like(state),
                                                                                     torch.zeros_like(lst_action))
                mu, std, act, logp, self.sample_hidden_state = self.inference_rnn_fix_one_action(state, lst_action)
                mu, std, act, logp = map(lambda x: x[:1].reshape((1, -1)), [mu, std, act, logp])
                # self.ep_tensor =
        if deterministic:
            return mu
        return act

    def inference_reset_one_hidden(self, idx):
        if self.rnn_fix_length is not None and self.rnn_fix_length > 0:
            raise NotImplementedError('if rnn fix length is set, parallel sampling is not allowed!!!')
        for i in range(len(self.sample_hidden_state)):
            if isinstance(self.sample_hidden_state[i], tuple):
                self.sample_hidden_state[i][0][0, idx] = 0
                self.sample_hidden_state[i][1][0, idx] = 0
            else:
                self.sample_hidden_state[i][0, idx] = 0

    @staticmethod
    def slice_tensor(x, slice_num):
        assert len(x.shape) == 3, 'slice operation should be added on 3-dim tensor'
        assert x.shape[1] % slice_num == 0, f'cannot reshape length with {x.shape[1]} to {slice_num} slices'
        s = x.shape
        x = x.reshape([s[0], s[1] // slice_num, slice_num, s[2]]).transpose(0, 1)
        return x

    @staticmethod
    def slice_tensor_overlap(x, slice_num):
        x_shape = x.shape
        x = torch.cat((torch.zeros((x_shape[0], slice_num-1, x_shape[2]), device=x.device), x), dim=1)
        xs = []
        for i in range(x_shape[1]):
            xs.append(x[:, i: i + slice_num, :])
        x = torch.cat(xs, dim=0)
        return x

    def generate_hidden_state_with_slice(self, sliced_state: torch.Tensor, sliced_lst_action: torch.Tensor):
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
                _, hidden_state_now = self.meta_forward(sliced_state[i], sliced_lst_action[i], hidden_state_now)
        return hidden_states

    @staticmethod
    def reshaping_hidden(full_hidden, init_hidden, slice_num, traj_len):
        for i in range(len(full_hidden)):
            # print(f'{hidden_state_now[i].shape}, {full_hidden[i].shape}')
            full_hidden[i] = torch.cat((init_hidden[i].squeeze(0).unsqueeze(1), full_hidden[i]), dim=1)
            # full_hidden[i] = full_hidden[i].unsqueeze(0)
        idx = [i * slice_num for i in range(traj_len // slice_num)]
        hidden_states = [item[:, idx].transpose(0, 1) for item in full_hidden]
        hidden_states_res = []
        for item in hidden_states:
            it_shape = item.shape
            hidden_states_res.append(item.reshape((1, it_shape[0] * it_shape[1], it_shape[2])))
        return hidden_states_res

    def generate_hidden_state(self, state: torch.Tensor, lst_action: torch.Tensor, slice_num, use_tmp_ep=False):
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
            if not use_tmp_ep:
                _, _, full_hidden = self.meta_forward(state, lst_action, hidden_state_now, require_full_output=True)
            else:
                _, _, full_hidden = self.get_ep_temp(torch.cat((state, lst_action), -1), hidden_state_now,
                                                     require_full_output=True)
            hidden_states_res = self.reshaping_hidden(full_hidden, hidden_state_now, slice_num, state.shape[1])
        return hidden_states_res

    @staticmethod
    def merge_slice_tensor(*args):
        res = []
        for item in args:
            s = item.shape
            # print(s)
            res.append(item.reshape(s[0] * s[1], s[2], s[3]))
        return res

    @staticmethod
    def merge_slice_hidden(hidden_states):
        """
        usage: [state, lst_action], hidden = self.merge_slice(sliced_state, sliced_lst_action, hidden_states)
        :param args:
        :param hidden_states:
        :return:
        """
        res_hidden = []
        len_hidden = len(hidden_states[0])
        for i in range(len_hidden):
            h = [item[i] for item in hidden_states]
            hid = torch.cat(h, dim=1)
            res_hidden.append(hid)
        return res_hidden

    @staticmethod
    def hidden_state_sample(hidden_state, inds):
        res_hidden = []
        len_hidden = len(hidden_state)
        for i in range(len_hidden):
            h = hidden_state[i][:, inds]
            hid = h
            res_hidden.append(hid)
        return res_hidden

    @staticmethod
    def hidden_state_slice(hidden_state, start, end):
        res_hidden = []
        len_hidden = len(hidden_state)
        for i in range(len_hidden):
            h = hidden_state[i][:, start: end]
            hid = h
            res_hidden.append(hid)
        return res_hidden


    @staticmethod
    def hidden_state_mask(hidden_state, masks):
        res_hidden = []
        len_hidden = len(hidden_state)
        for i in range(len_hidden):
            h = hidden_state[i].squeeze(0)[masks].unsqueeze(0)
            hid = h
            res_hidden.append(hid)
        return res_hidden

    @staticmethod
    def hidden_detach(hidden_state):
        res_hidden = []
        len_hidden = len(hidden_state)
        for i in range(len_hidden):
            res_hidden.append(hidden_state[i].detach())
        return res_hidden

    def _test_forward_time(self, device, num=1000, batch_size=1):

        h = self.make_init_state(batch_size, device)
        start_time = time.time()
        action_tensor = torch.randn((batch_size, 1, self.act_dim), device=device)
        obs_tensor = torch.randn((batch_size, 1, self.obs_dim), device=device)

        for i in range(num):
            self.forward(obs_tensor, action_tensor, h=h)
        end_time = time.time()
        print('pure forward time: {}, pure meta forward time: {}'.format(self.ep.cumulative_forward_time + self.up.cumulative_forward_time,
                                                                         self.ep.cumulative_meta_forward_time + self.up.cumulative_meta_forward_time))
        print('running for {} times, spending time is {:.2f}, batch size is {}, device: {}'.format(num, end_time - start_time, batch_size, device))
        self.ep.cumulative_forward_time = 0
        self.up.cumulative_forward_time = 0
        self.ep.cumulative_meta_forward_time = 0
        self.up.cumulative_meta_forward_time = 0

    def _test_inference_time(self, device, num=1000, batch_size=1):
        if not self.inference_check_hidden(1):
            self.inference_init_hidden(1, device)
        # h = self.make_init_state(batch_size, device)
        start_time = time.time()
        obs_act = torch.randn((1, self.obs_dim+self.act_dim), device=device)
        for i in range(num):
            self.inference_one_step(obs_act)
            # self.forward(torch.randn((batch_size, 1, self.obs_dim), device=device), torch.randn((batch_size, 1, self.act_dim), device=device), h=h)
        end_time = time.time()
        print('pure forward time: {}, pure meta forward time: {}'.format(
            self.ep.cumulative_forward_time + self.up.cumulative_forward_time,
            self.ep.cumulative_meta_forward_time + self.up.cumulative_meta_forward_time))
        # print('pure forward time: {}'.format(self.ep.cumulative_forward_time + self.up.cumulative_forward_time))
        print('running for {} times, spending time is {:.2f}, batch size is {}, device: {}'.format(num, end_time - start_time, batch_size, device))
        self.ep.cumulative_forward_time = 0
        self.up.cumulative_forward_time = 0
        self.ep.cumulative_meta_forward_time = 0
        self.up.cumulative_meta_forward_time = 0
