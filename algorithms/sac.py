import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from log_util.logger import Logger
from models.policy import Policy
from models.value import Value
from models.transition import Trainsition
from parameter.private_config import *
from agent.Agent import EnvRemoteArray
from envs.nonstationary_env import NonstationaryEnv
from utils.replay_memory import Memory, MemoryNp, MemoryArray
import gym
import torch
import numpy as np
import random
from utils.torch_utils import to_device
import time
from utils.timer import Timer
from algorithms.RMDM import RMDMLoss
from tqdm import tqdm
from utils.visualize_repre import visualize_repre, visualize_repre_real_param
import cProfile as profile
from envs.grid_world import RandomGridWorld
from envs.grid_world_general import RandomGridWorldPlat
from algorithms.contrastive import ContrastiveLoss

class SAC:
    def __init__(self):
        self.logger = Logger()
        # self.logger.set_tb_x_label('TotalInteraction')
        self.timer = Timer()
        self.parameter = self.logger.parameter
        self.policy_config = Policy.make_config_from_param(self.parameter)
        self.value_config = Value.make_config_from_param(self.parameter)
        self.transition_config = Trainsition.make_config_from_param(self.parameter)
        self.transition_config['stop_pg_for_ep'] = False
        self.env = NonstationaryEnv(gym.make(self.parameter.env_name), log_scale_limit=self.parameter.env_default_change_range,
                                    rand_params=self.parameter.varying_params)
        self.ood_env = NonstationaryEnv(gym.make(self.parameter.env_name), log_scale_limit=self.parameter.env_ood_change_range,
                                        rand_params=self.parameter.varying_params)
        self.global_seed(np.random, random, self.env, self.ood_env, seed=self.parameter.seed)
        torch.manual_seed(seed=self.parameter.seed)
        self.env_tasks = self.env.sample_tasks(self.parameter.task_num)
        self.test_tasks = self.env.sample_tasks(self.parameter.test_task_num)
        self.ood_tasks = self.ood_env.sample_tasks(self.parameter.test_task_num)
        self.training_agent = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                                             worker_num=1, seed=self.parameter.seed,
                                             deterministic=False, use_remote=False, policy_type=Policy,
                                             history_len=self.parameter.history_length, env_decoration=NonstationaryEnv,
                                             env_tasks=self.env_tasks,
                                             use_true_parameter=self.parameter.use_true_parameter,
                                             non_stationary=False)

        self.test_agent = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                                         worker_num=self.parameter.num_threads, seed=self.parameter.seed + 1,
                                         deterministic=True, use_remote=True, policy_type=Policy,
                                         history_len=self.parameter.history_length, env_decoration=NonstationaryEnv,
                                         env_tasks=self.test_tasks,
                                         use_true_parameter=self.parameter.use_true_parameter,
                                         non_stationary=False)

        self.ood_agent = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                                        worker_num=self.parameter.num_threads, seed=self.parameter.seed + 2,
                                        deterministic=True, use_remote=True, policy_type=Policy,
                                        history_len=self.parameter.history_length, env_decoration=NonstationaryEnv,
                                        env_tasks=self.ood_tasks,
                                        use_true_parameter=self.parameter.use_true_parameter,
                                        non_stationary=False)

        self.ood_ns_agent = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                                        worker_num=self.parameter.num_threads, seed=self.parameter.seed + 3,
                                        deterministic=True, use_remote=True, policy_type=Policy,
                                        history_len=self.parameter.history_length, env_decoration=NonstationaryEnv,
                                        env_tasks=self.ood_tasks,
                                        use_true_parameter=self.parameter.use_true_parameter,
                                        non_stationary=True)

        self.non_station_agent = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                                                worker_num=self.parameter.num_threads, seed=self.parameter.seed + 3,
                                                deterministic=True, use_remote=True, policy_type=Policy,
                                                history_len=self.parameter.history_length,
                                                env_decoration=NonstationaryEnv, env_tasks=self.test_tasks,
                                                use_true_parameter=self.parameter.use_true_parameter,
                                                non_stationary=True)
        self.non_station_agent_single_thread = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                                                worker_num=1, seed=self.parameter.seed + 4,
                                                deterministic=True, use_remote=False, policy_type=Policy,
                                                history_len=self.parameter.history_length,
                                                env_decoration=NonstationaryEnv, env_tasks=self.test_tasks,
                                                use_true_parameter=self.parameter.use_true_parameter,
                                                non_stationary=True)
        if self.parameter.use_true_parameter:
            self.policy_config['ep_dim'] = self.training_agent.env_parameter_len
            self.value_config['ep_dim'] = self.training_agent.env_parameter_len
        self.policy_config['logger'] = self.logger
        self.value_config['logger'] = self.logger
        self.transition_config['logger'] = self.logger
        self.loaded_pretrain = not self.parameter.ep_pretrain_path_suffix == 'None'
        self.freeze_ep = False
        self.policy_config['freeze_ep'] = self.freeze_ep
        self.value_config['freeze_ep'] = self.freeze_ep
        self.transition_config['freeze_ep'] = self.freeze_ep
        self.obs_dim = self.training_agent.obs_dim
        self.act_dim = self.training_agent.act_dim
        self.policy = Policy(self.training_agent.obs_dim, self.training_agent.act_dim, **self.policy_config)
        self.policy_for_test = Policy(self.training_agent.obs_dim, self.training_agent.act_dim, **self.policy_config)
        self.policy_target = Policy(self.training_agent.obs_dim, self.training_agent.act_dim, **self.policy_config)

        self.value1 = Value(self.training_agent.obs_dim, self.training_agent.act_dim, **self.value_config)
        self.value2 = Value(self.training_agent.obs_dim, self.training_agent.act_dim, **self.value_config)
        self.target_value1 = Value(self.training_agent.obs_dim, self.training_agent.act_dim, **self.value_config)
        self.target_value2 = Value(self.training_agent.obs_dim, self.training_agent.act_dim, **self.value_config)

        if not self.parameter.ep_pretrain_path_suffix == 'None':
            pretrain_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(self.logger.model_output_dir)),
                                      '{}-{}'.format(self.parameter.env_name, self.parameter.ep_pretrain_path_suffix),
                                              'model'), 'environment_probe.pt')
            self.logger('load model from {}'.format(pretrain_path))
            _, _, _, _, _ = map(lambda x: x.load(pretrain_path, map_location=torch.device('cpu')), [self.policy.ep, self.value1.ep,
                                                                                    self.value2.ep, self.target_value1.ep,
                                                                                    self.target_value2.ep])

        self.tau = self.parameter.sac_tau
        self.target_entropy = -self.parameter.target_entropy_ratio * self.act_dim
        # self.replay_buffer = MemoryNp(self.parameter.uniform_sample, self.parameter.rnn_slice_num)
        self.replay_buffer = MemoryArray(self.parameter.rnn_slice_num, max_trajectory_num=5000, max_traj_step=1050, fix_length=self.parameter.rnn_fix_length)
        # self.replay_buffer.set_max_size(self.parameter.sac_replay_size)
        self.policy_parameter = [*self.policy.parameters(True)]
        self.policy_optimizer = torch.optim.Adam(self.policy_parameter, lr=self.parameter.learning_rate)
        self.value_parameter = [*self.value1.parameters(True)] + [*self.value2.parameters(True)]
        if self.parameter.stop_pg_for_ep:
            self.value_parameter = [*self.value1.up.parameters(True)] + [*self.value2.up.parameters(True)]
        self.value_optimizer = torch.optim.Adam(self.value_parameter,
                                                lr=self.parameter.value_learning_rate)
        self.device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
        self.logger.log(f"torch device is {self.device}")
        self.log_sac_alpha = (torch.ones((1)).to(torch.get_default_dtype()
                                         ) * np.log(self.parameter.sac_alpha)).to(self.device).requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam([self.log_sac_alpha], lr=1e-2)
        self.rmdm_loss = RMDMLoss(max_env_len=self.parameter.task_num, tau=self.parameter.rmdm_tau)
        to_device(self.device, self.policy, self.policy_for_test, self.value1, self.value2, self.target_value1, self.target_value2)
        if self.parameter.transition_learn_aux:
            self.transition = Trainsition(self.training_agent.obs_dim, self.training_agent.act_dim, **self.transition_config)
            self.transition_optimizer = torch.optim.Adam([*self.transition.parameters(True)],
                                                         lr=self.parameter.learning_rate)
            to_device(self.device, self.transition)
        else:
            self.transition = None
            self.transition_optimizer = None
        if self.parameter.use_contrastive:
            self.contrastive_loss = ContrastiveLoss(self.parameter.ep_dim, self.parameter.task_num, ep=self.policy_target.ep)
            to_device(self.device, self.contrastive_loss)
        else:
            self.contrastive_loss = None
        self.logger.log(f'size of parameter of policy: {len(self.policy_parameter)}, '
                        f'size of parameter of value: {len(self.value_parameter)}')
        self.all_repre = None
        self.all_valids = None
        self.all_repre_target = None
        self.all_valids_target = None
        self.all_tasks_validate = None
        self.log_consis_w_alpha = (torch.ones((1)).to(torch.get_default_dtype()
                                                      ) * np.log(self.parameter.consistency_loss_weight)).to(self.device).requires_grad_(
            True)
        self.log_diverse_w_alpha = (torch.ones((1)).to(torch.get_default_dtype()
                                                       ) * np.log(1.0)).to(self.device).requires_grad_(
            True)
        self.repre_loss_factor = self.parameter.repre_loss_factor
        self.w_optimizer = torch.optim.SGD([self.log_consis_w_alpha, self.log_diverse_w_alpha], lr=1e-1)
        self.w_log_max = 5.5
        self.env_param_dict = self.training_agent.make_env_param_dict_from_params(self.parameter.varying_params)
        self.logger.log('environment parameter dict: ')
        self.logger.log_dict_single(self.env_param_dict)

    @staticmethod
    def global_seed(*args, seed):
        for item in args:
            item.seed(seed)

    def sac_update(self, state, action, next_state, reward, mask, last_action, valid, task, env_param,
                   policy_hidden=None, value_hidden1=None, value_hidden2=None, transition_hidden=None,
                   can_optimize_ep=True):
        """reward, mask should be (-1, 1)"""
        alpha = self.log_sac_alpha.exp()
        if self.parameter.rnn_fix_length is None or self.parameter.rnn_fix_length == 0:
            valid_num = valid.sum()
        else:
            valid_num = state.shape[0]
        if self.contrastive_loss is not None:
            query_tensor = self.contrastive_loss.get_query_tensor(state, last_action)
        else:
            query_tensor = None
        # if not FC_MODE:
        rmdm_loss_tensor = consistency_loss = diverse_loss = None
        batch_task_num = 1
        """update critic/value net"""
        self.timer.register_point('calculating_target_Q', level=3)     # (TIME: 0.011)
        consis_w = torch.exp(self.log_consis_w_alpha)
        diverse_w = torch.exp(self.log_diverse_w_alpha)
        with torch.no_grad():
            state_shape = state.shape
            state_dim = len(state_shape)
            if self.parameter.rnn_fix_length is None or self.parameter.rnn_fix_length == 0:
                total_state = torch.cat((state[..., :1, :], next_state), state_dim - 2)
                total_last_action = torch.cat((last_action, action[..., -1:, :]), state_dim - 2)
            else:
                total_state = next_state
                total_last_action = action
            _, _, next_action_total, nextact_logprob, _ = self.policy.rsample(total_state, total_last_action, policy_hidden)
            ep_total = self.policy.ep_tensor.detach() if self.parameter.stop_pg_for_ep else None
            if self.parameter.stop_pg_for_ep and self.rmdm_loss.history_env_mean is not None:
                ind_ = torch.abs(task[..., -1, 0]-1).to(dtype=torch.int64)
                ep_total = self.rmdm_loss.history_env_mean[ind_].detach()
                # ep_total = torch.cat([ep_total] * state.shape[1], dim=1)
                # ep_total = env_param[:, :, -2:]
            if self.parameter.rnn_fix_length and self.parameter.stop_pg_for_ep:
                target_Q1, _ = self.target_value1.forward(total_state[:, -1:, :], total_last_action[:, -1:, :],
                                                          next_action_total[:, -1:, :], value_hidden1, ep_out=ep_total[:, -1:, :])
                target_Q2, _ = self.target_value2.forward(total_state[:, -1:, :], total_last_action[:, -1:, :],
                                                          next_action_total[:, -1:, :], value_hidden2, ep_out=ep_total[:, -1:, :])
            else:
                target_Q1, _ = self.target_value1.forward(total_state, total_last_action, next_action_total, value_hidden1, ep_out=ep_total)
                target_Q2, _ = self.target_value2.forward(total_state, total_last_action, next_action_total, value_hidden2, ep_out=ep_total)

            if self.parameter.rnn_fix_length and not self.parameter.stop_pg_for_ep:
                target_Q1, target_Q2 = target_Q1[..., -1:, :], target_Q2[..., -1:, :]

            if self.parameter.rnn_fix_length is None or self.parameter.rnn_fix_length == 0:
                target_Q1, target_Q2, nextact_logprob = target_Q1[..., 1:, :], target_Q2[..., 1:, :], \
                                                        nextact_logprob[..., 1:, :]
                target_v = torch.min(target_Q1, target_Q2) - alpha.detach() * nextact_logprob
                target_Q = (reward + (mask * self.parameter.gamma * target_v)).detach()
            else:
                nextact_logprob = nextact_logprob[..., -1:, :]
                target_v = torch.min(target_Q1, target_Q2) - alpha.detach() * nextact_logprob
                target_Q = (reward[..., -1:, :] + (mask[..., -1:, :] * self.parameter.gamma * target_v)).detach()
        self.timer.register_end(level=3)
        self.timer.register_point('current_Q', level=3)     # (TIME: 0.006)
        _, _, action_rsample, logprob, _ = self.policy.rsample(state, last_action, policy_hidden)
        ep = self.policy.ep_tensor
        ep_current = self.policy.ep_tensor.detach() if self.parameter.stop_pg_for_ep else None
        if self.parameter.stop_pg_for_ep and self.rmdm_loss.history_env_mean is not None:
            ep_current = self.rmdm_loss.history_env_mean[torch.abs(task[..., -1, 0] - 1).to(dtype=torch.int64)].detach()
            # ep_current = torch.cat([ep_current] * state.shape[1], dim=1)
            # ep_current = env_param[:, :, -2:]
        if self.parameter.rnn_fix_length and self.parameter.stop_pg_for_ep:
            current_Q1, _ = self.value1.forward(state[:, -1:, :], last_action[:, -1:, :], action[:, -1:, :], value_hidden1, ep_out=ep_current[:, -1:, :])
            current_Q2, _ = self.value2.forward(state[:, -1:, :], last_action[:, -1:, :], action[:, -1:, :], value_hidden2, ep_out=ep_current[:, -1:, :])
        elif self.parameter.rnn_fix_length:
            current_Q1, _ = self.value1.forward(state, last_action, action, value_hidden1, ep_out=ep_current)
            current_Q2, _ = self.value2.forward(state, last_action, action, value_hidden2, ep_out=ep_current)
            current_Q1, current_Q2 = current_Q1[:, -1:, :], current_Q2[:, -1:, :]
        else:
            current_Q1, _ = self.value1.forward(state, last_action, action, value_hidden1, ep_out=ep_current)
            current_Q2, _ = self.value2.forward(state, last_action, action, value_hidden2, ep_out=ep_current)

        self.timer.register_end(level=3)
        if self.parameter.rnn_fix_length is None or self.parameter.rnn_fix_length == 0:
            q1_loss, q2_loss = map(lambda c: ((c - target_Q) * valid).pow(2).sum() / valid_num, [current_Q1, current_Q2])
        else:
            q1_loss = (current_Q1 - target_Q).pow(2).mean()
            q2_loss = (current_Q2 - target_Q).pow(2).mean()
        critic_loss = q1_loss + q2_loss

        self.timer.register_point('value_optimization', level=3)     # (TIME: 0.028)
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        if torch.isnan(critic_loss).any().item():
            self.logger.log(f"nan found in critic loss, state: {state.abs().sum()}, "
                            f"last action: {last_action.abs().sum()}, "
                            f"action: {action.abs().sum()}")
            return None
        self.value_optimizer.step()
        self.timer.register_end(level=3)
        """update policy and alpha"""
        self.timer.register_point('actor_loss', level=3)     # (TIME: 0.012)

        if self.parameter.ep_smooth_factor > 0:
            ep = self.policy.tmp_ep_res(state, last_action, policy_hidden)
        if self.parameter.rnn_fix_length and self.parameter.stop_pg_for_ep:
            actor_q1, _ = self.value1.forward(state[:, -1:, :], last_action[:, -1:, :], action_rsample[:, -1:, :], value_hidden1, ep_out=ep_current[:, -1:, :])
            actor_q2, _ = self.value2.forward(state[:, -1:, :], last_action[:, -1:, :], action_rsample[:, -1:, :], value_hidden2, ep_out=ep_current[:, -1:, :])
        else:
            actor_q1, _ = self.value1.forward(state, last_action, action_rsample, value_hidden1, ep_out=ep_current)
            actor_q2, _ = self.value2.forward(state, last_action, action_rsample, value_hidden2, ep_out=ep_current)
        if self.parameter.rnn_fix_length:
            actor_q1, actor_q2, logprob = map(lambda x: x[..., -1:, :],
                                                     [actor_q1, actor_q2, logprob])
        actor_q = torch.min(actor_q1, actor_q2)
        if self.parameter.rnn_fix_length:
            actor_loss = (alpha.detach() * logprob - actor_q).mean()
        else:
            actor_loss = ((alpha.detach() * logprob - actor_q) * valid).sum() / valid_num
        if self.parameter.use_rmdm and not self.parameter.share_ep and not self.freeze_ep and can_optimize_ep:
            self.timer.register_point('rmdm_loss', level=5)
            # ep, _ = self.policy.get_ep_temp(torch.cat((state, last_action), -1), policy_hidden)
            if self.parameter.rnn_fix_length:
                ep = ep[..., -1:, :]

            if self.parameter.rnn_fix_length:
                rmdm_loss_tensor, consistency_loss, diverse_loss, batch_task_num, consis_w_loss, diverse_w_loss, \
                    all_repre, all_valids = self.rmdm_loss.rmdm_loss_timing(ep, task, valid, consis_w, diverse_w,
                                                                            True, True,
                                                                            rbf_radius=self.parameter.rbf_radius,)
            else:
                rmdm_loss_tensor, consistency_loss, diverse_loss, batch_task_num, consis_w_loss, diverse_w_loss, \
                    all_repre, all_valids = self.rmdm_loss.rmdm_loss(ep, task, valid, consis_w, diverse_w, True,
                                                                        True, rbf_radius=self.parameter.rbf_radius)
            self.all_repre = [item.detach() for item in all_repre]
            self.all_valids = [item.detach() for item in all_valids]
            self.all_tasks = self.rmdm_loss.lst_tasks
            do_not_train_ep = False
            if self.replay_buffer.size < self.parameter.minimal_repre_rp_size\
                    or len(self.all_tasks) < int(0.5 * self.parameter.task_num):
                do_not_train_ep = True
            if rmdm_loss_tensor is not None and not do_not_train_ep:
                if torch.isnan(consistency_loss).any().item() or torch.isnan(diverse_loss).any().item():
                    self.logger.log(f'rmdm produce nan: consistency: {consistency_loss.item()}, '
                                    f'diverse loss: {diverse_loss.item()}')
                actor_loss = actor_loss + rmdm_loss_tensor * self.repre_loss_factor
                if self.parameter.l2_norm_for_ep > 0.0 :
                    l2_norm_for_ep = 0
                    ep = self.policy.ep if self.parameter.ep_smooth_factor == 0.0 else self.policy.ep_temp
                    for parameter_ in ep.parameters(True): # 8
                        l2_norm_for_ep = l2_norm_for_ep + torch.norm(parameter_).pow(2)
                    actor_loss = actor_loss + l2_norm_for_ep* self.parameter.l2_norm_for_ep
            else:
                pass
            self.timer.register_end(level=5)
        else:
            self.timer.register_point('rmdm_loss', level=5)
            self.timer.register_end(level=5)

        uposi_loss = None
        if self.parameter.use_uposi and not self.parameter.share_ep and not self.freeze_ep:
            target_ep_output = env_param
            if self.parameter.rnn_fix_length:
                target_ep_output = target_ep_output[..., -1:, -2:]
                ep = ep[..., -1:, :]
                uposi_loss = (ep - target_ep_output).pow(2).mean()
            else:
                uposi_loss = ((ep - target_ep_output[..., -2:]).pow(2) * valid).sum() / valid_num
            actor_loss = actor_loss + uposi_loss
            pass
        transition_loss = None
        if self.parameter.transition_learn_aux and not self.parameter.share_ep and not self.freeze_ep:
            if self.parameter.rnn_fix_length:
                ep = ep[..., -1:, :]
                transition_loss = (self.transition.forward(state[:, -1:, :], last_action[:, -1:, :], action[:, -1:, :],
                                           transition_hidden, ep_out=ep[:, -1:, :])[0] - next_state).pow(2).mean()
            else:
                transition_loss = ((self.transition.forward(state, last_action, action, transition_hidden, ep_out=ep)[0]
                                    - next_state).pow(2) * valid).sum() / valid_num

            actor_loss = actor_loss + transition_loss
            pass
        contrastive_loss = None
        if self.parameter.use_contrastive and not self.parameter.share_ep and not self.freeze_ep:
            if self.parameter.rnn_fix_length:
                ep = ep[..., -1:, :]
                contrastive_loss = self.contrastive_loss.contrastive_loss(ep, query_tensor, task)
                actor_loss = actor_loss + contrastive_loss
        self.timer.register_point('policy_optimization', level=3)     # (TIME: 0.026)
        self.policy_optimizer.zero_grad()
        if self.parameter.transition_learn_aux:
            self.transition_optimizer.zero_grad()
        if torch.isnan(actor_loss).any().item():
            self.logger.log(f"nan found in actor loss, state: {state.abs().sum()}, "
                            f"last action: {last_action.abs().sum()}, "
                            f"action: {action.abs().sum()}")
            return None
        actor_loss.backward()
        self.policy_optimizer.step()
        if self.parameter.transition_learn_aux:
            self.transition_optimizer.step()
        if self.parameter.ep_smooth_factor > 0:
            self.policy.apply_temp_ep(self.parameter.ep_smooth_factor)
        self.timer.register_end(level=3)
        if self.parameter.rnn_fix_length:
            alpha_loss = - (alpha * ((logprob + self.target_entropy)).detach()).mean()
        else:
            alpha_loss = - (alpha * ((logprob + self.target_entropy) * valid).detach()).sum() / valid_num
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_sac_alpha.clamp_max_(0)
        w_loss = 0
        with torch.no_grad():
            self.log_diverse_w_alpha.clamp_max_(self.w_log_max)
            self.log_consis_w_alpha.clamp_max_(self.w_log_max)
        self.value_function_soft_update()
        if self.parameter.share_ep:
            self.policy.ep.copy_weight_from(self.value1.ep, tau=0.0)
        if self.parameter.rnn_fix_length:
            q_mean1 = actor_q1.mean().item()
            q_mean2 = actor_q2.mean().item()
            logp_pi = logprob.mean().item()

        else:
            q_mean1 = ((actor_q1 * valid).sum() / valid_num).item()
            q_mean2 = ((actor_q2 * valid).sum() / valid_num).item()
            logp_pi = ((logprob * valid).sum() / valid_num).item()
        if self.parameter.use_rmdm:
            rmdm_loss_float = rmdm_loss_tensor.item() if rmdm_loss_tensor is not None else 0
            consistency_loss_float = consistency_loss.item() if consistency_loss is not None else 0
            diverse_loss_float = diverse_loss.item() if diverse_loss is not None else 0
        else:
            rmdm_loss_float = 0.0
            consistency_loss_float = 0.0
            diverse_loss_float = 0.0
            batch_task_num = 0
        if self.contrastive_loss is not None:
            self.contrastive_loss.update_ep(self.policy.ep)
        return dict(
            CriticLoss=critic_loss.item(),
            ActorLoss=actor_loss.item(),
            Alpha=alpha.item(),
            QMean1=q_mean1,
            QMean2=q_mean2,
            Logp=logp_pi,
            PolicyGradient=0,
            ValueGradient=0,
            rmdmLoss=rmdm_loss_float,
            ConsistencyLoss=consistency_loss_float,
            DiverseLoss=diverse_loss_float,
            BatchTaskNum=batch_task_num,
            w_loss=w_loss.item() if isinstance(w_loss, torch.Tensor) else w_loss,
            consistency_w=consis_w.item(),
            diverse_w=diverse_w.item(),
            UPOSILoss=0 if uposi_loss is None else uposi_loss.item(),
            TransitionLoss=0 if transition_loss is None else transition_loss.item(),
            ContrastiveLoss=0 if contrastive_loss is None else contrastive_loss.item(),
        )

    def sac_update_from_buffer(self):
        log = {}
        for _ in range(self.parameter.update_interval):
            self.timer.register_point('sample_from_replay', level=1)     # (TIME: 0.4)
            if FC_MODE:
                batch = self.replay_buffer.sample_transitions(self.parameter.sac_mini_batch_size)
            else:
                if self.parameter.rnn_fix_length:
                    batch = self.replay_buffer.sample_fix_length_sub_trajs(self.parameter.sac_mini_batch_size,
                                                                           self.parameter.rnn_fix_length)
                else:
                    batch, total_size = self.replay_buffer.sample_trajs(self.parameter.sac_mini_batch_size,
                                                        self.parameter.rnn_sample_max_batch_size)
                # self.logger.log(f'total transition in the trajectories is {total_size}, state shape: {np.array(batch.state).shape}')

            dtype = torch.get_default_dtype()
            device = self.device
            self.timer.register_point('from_numpy', level=1)     # (TIME: 0.4)
            # self.logger(np.array(batch.state), np.array(batch.reward))
            states, next_states, actions, last_action, rewards, masks, valid, task, env_param = \
                    map(lambda x: torch.from_numpy(np.array(x)).to(dtype=dtype, device=device),
                    [batch.state, batch.next_state, batch.action, batch.last_action,
                        batch.reward, batch.mask, batch.valid, batch.task, batch.env_param])
            self.timer.register_end(level=1)
            if not FC_MODE:
                self.timer.register_point('making_slice', level=3)     # (TIME: 0.14)
                if self.parameter.rnn_fix_length is None or self.parameter.rnn_fix_length == 0:
                    self.timer.register_point('generate_hidden_state', level=4)
                    hidden_policy = self.policy.generate_hidden_state(states, last_action,
                                                                      slice_num=self.parameter.rnn_slice_num)
                    hidden_value1 = self.value1.generate_hidden_state(states, last_action, actions,
                                                                      slice_num=self.parameter.rnn_slice_num)
                    hidden_value2 = self.value2.generate_hidden_state(states, last_action, actions,
                                                                      slice_num=self.parameter.rnn_slice_num)
                    if self.parameter.transition_learn_aux:
                        hidden_transition = self.transition.generate_hidden_state(states, last_action, actions,
                                                                                  slice_num=self.parameter.rnn_slice_num)
                    self.timer.register_point('Policy.slice_tensor', level=4)
                    states, next_states, actions, last_action, rewards, masks, valid, task, env_param = \
                        map(Policy.slice_tensor, [states, next_states, actions, last_action, rewards, masks, valid, task, env_param],
                            [self.parameter.rnn_slice_num] * 9)

                    self.timer.register_point('Policy.merge_slice_tensor', level=4)
                    states, next_states, actions, last_action, rewards, masks, valid, task, env_param = Policy.merge_slice_tensor(
                        states, next_states, actions, last_action, rewards, masks, valid, task, env_param
                    )
                    mask_for_valid = valid.sum(dim=-2, keepdim=True)[..., 0, 0] > 0
                    states, next_states, actions, last_action, rewards, masks, valid, task, env_param = map(
                        lambda x: x[mask_for_valid],
                        [states, next_states, actions, last_action, rewards, masks, valid, task, env_param]
                    )
                    hidden_policy, hidden_value1, hidden_value2 = map(
                        Policy.hidden_state_mask,
                        [hidden_policy, hidden_value1, hidden_value2],
                        [mask_for_valid, mask_for_valid, mask_for_valid]
                    )
                    minibatch_size = self.parameter.sac_mini_batch_size
                    traj_num = min(max(minibatch_size // self.parameter.rnn_slice_num, 1), states.shape[0])
                    total_inds = np.random.permutation(states.shape[0]).tolist()[:traj_num]
                    hidden_policy, hidden_value1, hidden_value2 = \
                        map(Policy.hidden_state_sample,
                            [hidden_policy, hidden_value1, hidden_value2],
                            [total_inds, total_inds, total_inds])
                    states, next_states, actions, last_action, rewards, masks, valid, task, env_param = \
                        map(lambda x: x[total_inds],
                            [states, next_states, actions, last_action, rewards, masks, valid, task, env_param])
                    self.timer.register_end(level=4)
                    if self.parameter.transition_learn_aux:

                        hidden_transition = Policy.hidden_state_mask(hidden_transition, mask_for_valid)
                        hidden_transition = Policy.hidden_state_sample(hidden_transition, total_inds)

                else:
                    self.timer.register_point('Policy.slice_tensor', level=4)     # (TIME: 0.132)
                    # states, next_states, actions, last_action, rewards, masks, valid, task = \
                    #     map(Policy.slice_tensor_overlap, [states, next_states, actions, last_action, rewards, masks, valid, task],
                    #         [self.parameter.rnn_fix_length] * 8)
                    self.timer.register_end(level=4)
                    # mask_for_valid = valid[..., -1, 0] == 1
                    # states, next_states, actions, last_action, rewards, masks, valid, task = \
                    #     map(lambda x: x[mask_for_valid], [states, next_states, actions, last_action,
                    #                                       rewards, masks, valid, task])
                    self.timer.register_point('generate_hidden_state', level=4)
                    hidden_policy = self.policy.make_init_state(batch_size=states.shape[0], device=states.device)
                    hidden_value1 = self.value1.make_init_state(batch_size=states.shape[0], device=states.device)
                    hidden_value2 = self.value2.make_init_state(batch_size=states.shape[0], device=states.device)
                    if self.parameter.transition_learn_aux:
                        hidden_transition = self.transition.make_init_state(batch_size=states.shape[0], device=states.device)
                    self.timer.register_end(level=4)
                self.timer.register_end(level=3)
            else:
                hidden_policy, hidden_value1, hidden_value2 = [], [], []
            self.timer.register_end(level=1)
            #if custom_additional_reward is not None:
            #    with torch.no_grad():
            #        rewards = rewards + custom_additional_reward(states)
            states, next_states, actions, last_action, rewards, masks, valid, task, env_param = map(
                lambda x: x.detach(),
                [states, next_states, actions, last_action, rewards, masks, valid, task, env_param]
            )
            hidden_policy, hidden_value1, hidden_value2 = map(
                Policy.hidden_detach,
                [hidden_policy, hidden_value1, hidden_value2]
            )
            if self.parameter.transition_learn_aux:
                hidden_transition = Policy.hidden_detach(hidden_transition)
            with torch.set_grad_enabled(True):
                if FC_MODE:
                    self.timer.register_point('self.sac_update', level=1)
                    res_dict = self.sac_update(states, actions, next_states,
                                               rewards, masks, last_action, valid, task, env_param, hidden_policy,
                                               hidden_value1, hidden_value2)
                    self.timer.register_end(level=1)
                else:
                    point_num = states.shape[0]
                    total_inds = np.random.permutation(point_num).tolist()
                    iter_batch_size = states.shape[0] // self.parameter.sac_inner_iter_num
                    # self.logger(f'valid traj num: {states.shape[0]}, batch size: {iter_batch_size}')
                    for i in range(self.parameter.sac_inner_iter_num):
                        self.timer.register_point('sample_from_batch', level=1)     # (TIME: 0.003)
                        start = i * iter_batch_size
                        end = min((i+1) * iter_batch_size, states.shape[0])
                        states_batch, next_states_batch, actions_batch, \
                        last_action_batch, rewards_batch, masks_batch, valid_batch, task_batch, env_param_batch = \
                        map(lambda x: x[start: end], [
                            states, next_states, actions, last_action, rewards, masks, valid, task, env_param
                        ])
                        data_is_valid = False
                        if self.parameter.rnn_fix_length:
                            if valid_batch[..., -1:, :].sum().item() >= 2:
                                data_is_valid = True
                        elif valid_batch.sum().item() >= 2:
                            data_is_valid = True
                        if not data_is_valid:
                            print('data is not valid!!')
                            continue
                        hidden_policy_batch, hidden_value1_batch, hidden_value2_batch = \
                            map(Policy.hidden_state_slice,
                                [hidden_policy, hidden_value1, hidden_value2],
                                [start] * 3,
                                [end] * 3)
                        if self.parameter.transition_learn_aux:
                            hidden_transition_batch = Policy.hidden_state_slice(hidden_transition, start, end)
                        else:
                            hidden_transition_batch = None
                        self.timer.register_point('self.sac_update', level=1)     # (TIME: 0.091)
                        can_optimize_ep = self.replay_buffer.size > self.parameter.ep_start_num
                        res_dict = self.sac_update(states_batch, actions_batch, next_states_batch, rewards_batch,
                                                   masks_batch, last_action_batch, valid_batch, task_batch, env_param_batch,
                                                   hidden_policy_batch, hidden_value1_batch, hidden_value2_batch,
                                                   hidden_transition_batch, can_optimize_ep)
                        self.timer.register_end(level=1)
                if res_dict is not None:
                    for key in res_dict:
                        if key in log:
                            log[key].append(res_dict[key])
                        else:
                            log[key] = [res_dict[key]]
        return log

    def update(self):
        if self.parameter.bottle_neck:
            self.policy.set_deterministic_ep(deterministic=False)
        log = self.sac_update_from_buffer()
        # self.policy.set_deterministic_ep(deterministic=True)

        return log

    @staticmethod
    def append_key(d, tail):
        res = {}
        for k, v in d.items():
            res[k+tail] = v
        return res

    def value_function_soft_update(self):
        if self.parameter.stop_pg_for_ep:
            self.target_value1.up.copy_weight_from(self.value1.up, self.tau)
            self.target_value2.up.copy_weight_from(self.value2.up, self.tau)
        else:
            self.target_value1.copy_weight_from(self.value1, self.tau)
            self.target_value2.copy_weight_from(self.value2, self.tau)
    def run(self):
        total_steps = 0
        if self.replay_buffer.size < self.parameter.start_train_num:
            self.policy.to(device=torch.device('cpu'))
            self.logger(f"init samples!!!")
            while self.replay_buffer.size <= self.parameter.start_train_num:
                mem, log = self.training_agent.sample1step(self.policy,
                                                           self.replay_buffer.size < self.parameter.random_num,
                                                           device=torch.device('cpu'))
                self.replay_buffer.mem_push(mem)
                total_steps += 1
            self.logger("init done!!!")
        for iter in range(self.parameter.max_iter_num):
            self.policy.to(torch.device('cpu'))
            future_test = self.test_agent.submit_task(self.parameter.test_sample_num, self.policy)
            future_ood = self.ood_agent.submit_task(self.parameter.test_sample_num, self.policy)
            future_ns = self.non_station_agent.submit_task(self.parameter.test_sample_num, self.policy)
            future_ood_ns = self.ood_ns_agent.submit_task(self.parameter.test_sample_num, self.policy)
            training_start = time.time()
            single_step_iterater = range(self.parameter.min_batch_size) if not USE_TQDM else\
                tqdm(range(self.parameter.min_batch_size))
            for step in single_step_iterater:
                # self.policy.to(torch.device('cpu'))
                self.policy.to(self.device)
                self.timer.register_point('sample1step')
                mem, log = self.training_agent.sample1step(self.policy,
                                                           self.replay_buffer.size < self.parameter.random_num,
                                                           device=self.device)
                self.timer.register_end()

                if step % (self.parameter.update_interval * self.parameter.sac_inner_iter_num) == 0 \
                        and self.replay_buffer.size > self.parameter.start_train_num and len(self.replay_buffer) > 1:
                    self.policy.to(self.device)
                    self.timer.register_point('self.update')
                    update_log = self.update()
                    self.timer.register_end()
                    log.update(update_log)
                    pass
                self.timer.register_point('self.replay_buffer.mem_push')
                self.replay_buffer.mem_push(mem)
                self.timer.register_end()
                self.logger.add_tabular_data(**log, tb_prefix='training')
            representation_behaviour, diff_from_expert = self.test_non_stationary_repre()
            if iter % 10 == 0:
                if self.all_repre is not None:
                    fig, fig_mean = visualize_repre(self.all_repre, self.all_valids,
                                                    os.path.join(self.logger.output_dir, 'visual.png'),
                                                    self.env_param_dict, self.all_tasks )
                    fig_real_param = visualize_repre_real_param(self.all_repre, self.all_valids, self.all_tasks,
                                                                self.env_param_dict)
                    if fig:
                        self.logger.tb.add_figure('figs/repre', fig, iter)
                        self.logger.tb.add_figure('figs/repre_mean', fig_mean, iter)
                        self.logger.tb.add_figure('figs/repre_real', fig_real_param, iter)
                self.logger.tb.add_figure('figs/policy_behaviour', representation_behaviour, iter)
            total_steps += self.parameter.min_batch_size
            training_end = time.time()
            self.logger.log('start testing...')
            batch_test, log_test, mem_test = self.test_agent.query_sample(future_test, need_memory=True)
            batch_ood, log_ood, mem_ood = self.ood_agent.query_sample(future_ood, need_memory=True)
            batch_ood_ns, log_ood_ns, mem_ood_ns = self.ood_ns_agent.query_sample(future_ood_ns, need_memory=True)
            batch_non_station, log_non_station, mem_non_station = self.non_station_agent.query_sample(future_ns,
                                                                                                      need_memory=True)
            testing_end = time.time()
            self.logger.add_tabular_data(tb_prefix='evaluation', **self.append_key(log_test, "Test"))
            self.logger.add_tabular_data(tb_prefix='evaluation', **self.append_key(log_ood, "OOD"))
            self.logger.add_tabular_data(tb_prefix='evaluation', **self.append_key(log_non_station, "NS"))
            self.logger.add_tabular_data(tb_prefix='evaluation', **self.append_key(log_ood_ns, 'OOD_NS'))
            self.logger.log_tabular('TotalInteraction', total_steps, tb_prefix='timestep')
            self.logger.log_tabular('OODDeltaVSTestRet', np.mean(log_ood['EpRet']) - np.mean(log_test['EpRet']), tb_prefix='evaluation')
            self.logger.log_tabular('NSDeltaVSTestRet', np.mean(log_non_station['EpRet']) - np.mean(log_test['EpRet']), tb_prefix='evaluation')
            self.logger.log_tabular('ReplayBufferTrajNum', len(self.replay_buffer), tb_prefix='timestep')
            self.logger.log_tabular('ReplayBufferSize', self.replay_buffer.size, tb_prefix='timestep')
            self.logger.log_tabular('TrainingPeriod', training_end - training_start, tb_prefix='period')
            self.logger.log_tabular('TestingPeriod', testing_end - training_end, tb_prefix='period')
            self.logger.log_tabular('DiffFromExpert', diff_from_expert[0], tb_prefix='performance')
            self.logger.log_tabular('AtTargetRatio', diff_from_expert[1], tb_prefix='performance')
            self.logger.add_tabular_data(tb_prefix='period', **self.timer.summary())
            self.logger.tb_header_dict['EpRet'] = 'performance'
            self.logger.tb_header_dict['EpRetOOD'] = 'performance'
            self.logger.tb_header_dict['EpRetOOD_NS'] = 'performance'
            self.logger.tb_header_dict['EpRetTest'] = 'performance'
            self.logger.tb_header_dict['EpRetNS'] = 'performance'
            self.logger.tb_header_dict['rmdmLoss'] = 'representations'
            self.logger.tb_header_dict['DiverseLoss'] = 'representations'
            self.logger.tb_header_dict['ConsistencyLoss'] = 'representations'
            if iter % 10 == 0:
                self.save()
            self.logger.dump_tabular()

    def save(self):
        self.policy.save(self.logger.model_output_dir)
        self.value1.save(self.logger.model_output_dir, 0)
        self.value2.save(self.logger.model_output_dir, 1)
        self.target_value1.save(self.logger.model_output_dir, "target0")
        self.target_value2.save(self.logger.model_output_dir, "target1")
        torch.save(self.policy_optimizer.state_dict(), os.path.join(self.logger.model_output_dir, 'policy_optim.pt'))
        torch.save(self.value_optimizer.state_dict(), os.path.join(self.logger.model_output_dir, 'value_optim.pt'))
        torch.save(self.alpha_optimizer.state_dict(), os.path.join(self.logger.model_output_dir, 'alpha_optim.pt'))

    def load(self):
        self.policy.load(self.logger.model_output_dir, map_location=self.device)
        self.value1.load(self.logger.model_output_dir, 0, map_location=self.device)
        self.value2.load(self.logger.model_output_dir, 1, map_location=self.device)
        self.target_value1.load(self.logger.model_output_dir, "target0", map_location=self.device)
        self.target_value2.load(self.logger.model_output_dir, "target1", map_location=self.device)
        self.policy_optimizer.load_state_dict(torch.load(os.path.join(self.logger.model_output_dir, 'policy_optim.pt'),
                                                         map_location=self.device))
        self.value_optimizer.load_state_dict(torch.load(os.path.join(self.logger.model_output_dir, 'value_optim.pt'),
                                                        map_location=self.device))
        self.alpha_optimizer.load_state_dict(torch.load(os.path.join(self.logger.model_output_dir, 'alpha_optim.pt'),
                                                        map_location=self.device))

    def test_non_stationary_repre(self):
        self.policy_for_test.ep.copy_weight_from(self.policy.ep, 0.0)
        self.policy_for_test.up.copy_weight_from(self.policy.up, 0.0)
        self.policy_for_test.to(torch.device('cpu'))
        if self.parameter.bottle_neck:
            self.policy_for_test.set_deterministic_ep(True)
        fig, diff_from_expert = self.get_figure(self.policy_for_test, self.non_station_agent_single_thread, 1000)
        self.policy_for_test.to(self.device)
        return fig, diff_from_expert

    @staticmethod
    def get_figure(policy, agent, step_num, title=''):
        fig = plt.figure(18)
        plt.cla()
        ep_traj = []
        real_param = []
        action_discrepancy = []
        keep_at_target = []
        done = False
        while not done:
            mem, log, info = agent.sample1step1env(policy, False, render=False, need_info=True)
            done = mem.memory[0].done[0]
        for i in range(step_num):
            mem, log, info = agent.sample1step1env(policy, False, render=False, need_info=True)
            real_param.append(mem.memory[0].env_param)
            ep_traj.append(policy.ep_tensor[:1, ...].squeeze().detach().cpu().numpy())
            if isinstance(info, dict) and 'action_discrepancy' in info and info['action_discrepancy'] is not None:
                action_discrepancy.append(np.array([info['action_discrepancy'][0],
                                                    info['action_discrepancy'][1]]))
                keep_at_target.append(1 if info['keep_at_target'] else 0)
        ep_traj = np.array(ep_traj)
        real_param = np.array(real_param)
        change_inds = np.where(np.abs(np.diff(real_param[:, -1])) > 0)[0] + 1
        # print(np.hstack((ep_traj, real_param)))
        plt.plot(ep_traj[:, 0], label='x')
        plt.plot(ep_traj[:, 1], label='y')
        plt.plot(real_param[:, -1], label='real')
        diff_from_expert = 0
        at_target_ratio = 0
        if len(action_discrepancy) > 0:
            action_discrepancy = np.array(action_discrepancy)
            abs_res = np.abs(action_discrepancy[:, 0]) / 3 + np.abs(action_discrepancy[:, 1]) / 3
            plt.plot(np.arange(action_discrepancy.shape[0]), abs_res, '-*', label='diff')
            plt.title('mean discrepancy: {:.3f}'.format(np.mean(abs_res)))
            diff_from_expert = np.mean(abs_res)
            at_target_ratio = np.mean(keep_at_target)
        else:
            plt.title(title)
        for ind in change_inds:
            plt.plot([ind, ind], [-1.1, 1.1], 'k--', alpha=0.2)
        plt.ylim(bottom=-1.1, top=1.1)
        plt.legend()
        return fig, (diff_from_expert, at_target_ratio)


if __name__ == '__main__':
    import ray
    ray.init()
    sac = SAC()
    sac.run()
