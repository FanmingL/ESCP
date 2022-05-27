import sys
import os
import gym
import ray
import torch
import numpy as np
import random
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.policy import Policy
from utils.replay_memory import Memory, MemoryNp
from utils.history_construct import HistoryConstructor
from parameter.Parameter import Parameter
from parameter.private_config import SKIP_MAX_LEN_DONE, NON_STATIONARY_PERIOD, NON_STATIONARY_INTERVAL
from log_util.logger import Logger
from envs.grid_world import RandomGridWorld
from envs.grid_world_general import RandomGridWorldPlat
from parameter.private_config import ENV_DEFAULT_CHANGE


class EnvWorker:
    def __init__(self, parameter: Parameter, env_name='Hopper-v2', seed=0, policy_type=Policy,
                 history_len=0, env_decoration=None, env_tasks=None,
                 use_true_parameter=False, non_stationary=False):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.fix_env_setting = False
        self.set_global_seed(seed)
        self.use_true_parameter = use_true_parameter
        self.non_stationary = non_stationary
        if env_decoration is not None:
            default_change_range = ENV_DEFAULT_CHANGE if not hasattr(parameter, 'env_default_change_range') \
                else parameter.env_default_change_range
            if not hasattr(parameter, 'env_default_change_range'):
                print('[WARN]: env_default_change_range does not appears in parameter!')
            self.env = env_decoration(self.env, log_scale_limit=default_change_range,
                                    rand_params=parameter.varying_params)
        self.observation_space = self.env.observation_space

        self.history_constructor = HistoryConstructor(history_len, self.observation_space.shape[0],
                                                      self.action_space.shape[0], need_lst_action=True)
        self.env_tasks = None
        self.task_ind = -1
        self.env.reset()
        if env_tasks is not None and isinstance(env_tasks, list) and len(env_tasks) > 0:
            self.env_tasks = env_tasks
            self.task_ind = random.randint(0, len(self.env_tasks) - 1)
            self.env.set_task(self.env_tasks[self.task_ind])
        policy_config = Policy.make_config_from_param(parameter)
        if use_true_parameter:
            policy_config['ep_dim'] = self.env.env_parameter_length
        self.policy = policy_type(obs_dim=self.observation_space.shape[0],
                                  act_dim=self.action_space.shape[0],
                                  **policy_config)
        self.policy.inference_init_hidden(1)
        self.bottle_neck = parameter.bottle_neck
        self.ep_len = 0
        self.ep_cumrew = 0
        self.history_len = history_len
        self.ep_len_list = []
        self.ep_cumrew_list = []
        self.ep_rew_list = []
        self.state = self.reset(None)
        self.state = self.extend_state(self.state)

    def set_weight(self, state_dict):
        self.policy.load_state_dict(state_dict)

    def set_global_seed(self, seed):
        import numpy as np
        import torch
        import random
        self.env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)

    def get_weight(self):
        return self.policy.state_dict()

    def set_fix_env_setting(self, fix_env_setting=True):
        self.fix_env_setting = fix_env_setting

    def change_env_param(self, set_env_ind=None):
        if self.fix_env_setting:
            self.env.set_task(self.env_tasks[self.task_ind])
            return
        if self.env_tasks is not None and len(self.env_tasks) > 0:
            self.task_ind = random.randint(0, len(self.env_tasks) - 1) if set_env_ind is None or set_env_ind >= \
                                                                          len(self.env_tasks) else set_env_ind
            self.env.set_task(self.env_tasks[self.task_ind])
            if self.non_stationary:
                another_task = random.randint(0, len(self.env_tasks) - 1)
                env_param_list = [self.env_tasks[self.task_ind]] + [self.env_tasks[random.randint(0, len(self.env_tasks)-1)] for _ in range(15)]
                self.env.set_nonstationary_para(env_param_list,
                                                NON_STATIONARY_PERIOD, NON_STATIONARY_INTERVAL)

    def sample(self, min_batch, deterministic=False, env_ind=None):
        step_count = 0
        mem = Memory()
        log = {'EpRet': [],
               'EpMeanRew': [],
               'EpLen': []}
        if deterministic and self.bottle_neck:
            self.policy.set_deterministic_ep(True)
        elif self.bottle_neck and not deterministic:
            self.policy.set_deterministic_ep(False)
        with torch.no_grad():
            while step_count < min_batch:
                state = self.reset(env_ind)
                state = self.extend_state(state)
                self.policy.inference_init_hidden(1)
                while True:
                    state_tensor = torch.from_numpy(state).to(torch.get_default_dtype()).unsqueeze(0)
                    action_tensor = self.policy.inference_one_step(state_tensor, deterministic)[0]
                    action = action_tensor.numpy()
                    self.before_apply_action(action)
                    next_state, reward, done, _ = self.env.step(self.env.denormalization(action))
                    if self.non_stationary:
                        self.env_param_vector = self.env.env_parameter_vector
                    next_state = self.extend_state(next_state)
                    mask = 0.0 if done else 1.0
                    if SKIP_MAX_LEN_DONE and done and self.env._elapsed_steps >= self.env._max_episode_steps:
                        mask = 1.0
                    mem.push(state[self.action_space.shape[0]:], action.astype(state.dtype), [mask],
                             next_state[self.action_space.shape[0]:], [reward],
                             None, [self.task_ind + 1], self.env_param_vector,
                             state[:self.action_space.shape[0]], [done], [1])
                    self.ep_cumrew += reward
                    self.ep_len += 1
                    step_count += 1
                    if done:
                        log['EpMeanRew'].append(self.ep_cumrew / self.ep_len)
                        log['EpLen'].append(self.ep_len)
                        log['EpRet'].append(self.ep_cumrew)
                        break
                    state = next_state
        mem.memory = [mem.memory[0]]
        return mem, log

    def get_current_state(self):
        return self.state

    def extend_state(self, state):
        state = self.history_constructor(state)
        if self.use_true_parameter:
            state = np.hstack([state, self.env_param_vector])
        return state

    def before_apply_action(self, action):
        self.history_constructor.update_action(action)

    def reset(self, env_ind=None):
        self.history_constructor.reset()
        self.change_env_param(env_ind)
        state = self.env.reset()
        self.env_param_vector = self.env.env_parameter_vector
        self.ep_len = 0
        self.ep_cumrew = 0
        return state

    def step(self, action, env_ind=None, render=False, need_info=False):
        self.before_apply_action(action)
        next_state, reward, done, info = self.env.step(self.env.denormalization(action))
        if render:
            self.env.render()
        if self.non_stationary:
            self.env_param_vector = self.env.env_parameter_vector
        current_env_step = self.env._elapsed_steps
        self.state = next_state = self.extend_state(next_state)
        self.ep_len += 1
        self.ep_cumrew += reward
        cur_task_ind = self.task_ind
        cur_env_param = self.env_param_vector
        if done:
            self.ep_len_list.append(self.ep_len)
            self.ep_cumrew_list.append(self.ep_cumrew)
            self.ep_rew_list.append(self.ep_cumrew / self.ep_len)
            state = self.reset(env_ind)
            self.state = self.extend_state(state)
        if need_info:
            return next_state, reward, done, self.state, cur_task_ind, cur_env_param, current_env_step, info
        return next_state, reward, done, self.state, cur_task_ind, cur_env_param, current_env_step

    def collect_result(self):
        ep_len_list = self.ep_len_list
        self.ep_len_list = []
        ep_cumrew_list = self.ep_cumrew_list
        self.ep_cumrew_list = []
        ep_rew_list = self.ep_rew_list
        self.ep_rew_list = []
        log = {
        'EpMeanRew': ep_rew_list,
        'EpLen': ep_len_list,
        'EpRet': ep_cumrew_list
        }
        return log


class EnvRemoteArray:
    def __init__(self, parameter, env_name, worker_num=2, seed=None,
                 deterministic=False, use_remote=True,
                 policy_type=Policy, history_len=0, env_decoration=None,
                 env_tasks=None, use_true_parameter=False, non_stationary=False):
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.action_space = self.env.action_space
        self.set_seed(seed)
        self.non_stationary = non_stationary
        self.env_tasks = env_tasks
        # if worker_num == 1:
        #     use_remote = False
        RemoteEnvWorker = ray.remote(EnvWorker) if use_remote else EnvWorker
        if use_remote:
            self.workers = [RemoteEnvWorker.remote(parameter, env_name, random.randint(0, 10000),
                                                   policy_type, history_len, env_decoration, env_tasks,
                                                   use_true_parameter, non_stationary) for _ in range(worker_num)]
        else:
            self.workers = [RemoteEnvWorker(parameter, env_name, random.randint(0, 10000),
                                            policy_type, history_len, env_decoration, env_tasks,
                                            use_true_parameter, non_stationary) for _ in range(worker_num)]

        if env_decoration is not None:
            default_change_range = ENV_DEFAULT_CHANGE if not hasattr(parameter, 'env_default_change_range') \
                else parameter.env_default_change_range
            if not hasattr(parameter, 'env_default_change_range'):
                print('[WARN]: env_default_change_range does not appears in parameter!')
            self.env = env_decoration(self.env, log_scale_limit=default_change_range,
                                    rand_params=parameter.varying_params)
        net_config = Policy.make_config_from_param(parameter)
        self.policy = Policy(self.env.observation_space.shape[0], self.env.action_space.shape[0], **net_config)
        self.worker_num = worker_num
        self.env_name = env_name

        self.env.reset()
        if isinstance(env_tasks, list) and len(env_tasks) > 0:
            self.env.set_task(random.choice(env_tasks))
        self.env_parameter_len = self.env.env_parameter_length
        self.running_state = None
        self.deterministic = deterministic
        self.use_remote = use_remote
        self.total_steps = 0

    def set_seed(self, seed):
        if seed is None:
            return
        import numpy as np
        import torch
        import random
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)
        self.env.seed(seed)

    def set_fix_env_setting(self, fix_env_setting=True):
        if self.use_remote:
            ray.get([worker.set_fix_env_setting.remote(fix_env_setting) for worker in self.workers])
        else:
            for worker in self.workers:
                worker.set_fix_env_setting(fix_env_setting)

    def submit_task(self, min_batch, policy=None, env_ind=None):
        assert not (policy is None and self.policy is None)
        cur_policy = policy if policy is not None else self.policy
        ray.get([worker.set_weight.remote(cur_policy.state_dict()) for worker in self.workers])

        min_batch_per_worker = min_batch // self.worker_num + 1
        futures = [worker.sample.remote(min_batch_per_worker, self.deterministic, env_ind)
                   for worker in self.workers]
        return futures

    def query_sample(self, futures, need_memory):
        mem_list_pre = ray.get(futures)
        mem = Memory()
        [mem.append(new_mem) for new_mem, _ in mem_list_pre]
        logs = {key: [] for key in mem_list_pre[0][1]}
        for key in logs:
            for _, item in mem_list_pre:
                logs[key] += item[key]
        logs['TotalSteps'] = len(mem)
        batch = self.extract_from_memory(mem)
        if need_memory:
            return batch, logs, mem
        return batch, logs

    # always use remote
    def sample(self, min_batch, policy=None, env_ind=None):
        assert not (policy is None and self.policy is None)
        cur_policy = policy if policy is not None else self.policy
        ray.get([worker.set_weight.remote(cur_policy.state_dict()) for worker in self.workers])

        min_batch_per_worker = min_batch // self.worker_num + 1
        futures = [worker.sample.remote(min_batch_per_worker, self.deterministic, env_ind)
                                for worker in self.workers]
        mem_list_pre = ray.get(futures)
        mem = Memory()
        [mem.append(new_mem) for new_mem, _ in mem_list_pre]
        logs = {key: [] for key in mem_list_pre[0][1]}
        for key in logs:
            for _, item in mem_list_pre:
                logs[key] += item[key]
        logs['TotalSteps'] = len(mem)
        return mem, logs

    def sample_locally(self, min_batch, policy=None, env_ind=None):
        assert not (policy is None and self.policy is None)
        cur_policy = policy if policy is not None else self.policy
        for worker in self.workers:
            worker.set_weight(cur_policy.state_dict())
        min_batch_per_worker = min_batch // self.worker_num + 1
        mem_list_pre = [worker.sample(min_batch_per_worker, self.deterministic, env_ind)
                                for worker in self.workers]
        mem = Memory()
        [mem.append(new_mem) for new_mem, _ in mem_list_pre]
        logs = {key: [] for key in mem_list_pre[0][1]}
        for key in logs:
            for _, item in mem_list_pre:
                logs[key] += item[key]
        logs['TotalSteps'] = len(mem)
        return mem, logs

    def sample1step(self, policy=None, random=False, device=torch.device('cpu'), env_ind=None):
        assert not (policy is None and self.policy is None)
        cur_policy = policy if policy is not None else self.policy
        if (not self.use_remote) and self.worker_num == 1:
            return self.sample1step1env(policy, random, device, env_ind)
        if not cur_policy.inference_check_hidden(self.worker_num):
            cur_policy.inference_init_hidden(self.worker_num, device)
        if self.use_remote:
            states = ray.get([worker.get_current_state.remote() for worker in self.workers])
        else:
            states = [worker.get_current_state() for worker in self.workers]

        states = np.array(states)
        with torch.no_grad():
            if random:
                actions = [self.env.normalization(self.action_space.sample()) for item in states]
            else:
                states_tensor = torch.from_numpy(states).to(torch.get_default_dtype()).to(device).unsqueeze(1)
                actions = cur_policy.inference_one_step(states_tensor, self.deterministic).to(torch.device('cpu')).squeeze(1).numpy()

        if self.use_remote:
            srd = ray.get([worker.step.remote(action, env_ind) for action, worker in zip(actions, self.workers)])
        else:
            srd = [worker.step(action) for action, worker in zip(actions, self.workers)]

        mem = Memory()
        for ind, (next_state, reward, done, _, task_ind, env_param, current_steps) in enumerate(srd):
            if done:
                cur_policy.inference_reset_one_hidden(ind)
            mask = 0.0 if done else 1.0
            if SKIP_MAX_LEN_DONE and done and current_steps >= self.env._max_episode_steps:
                mask = 1.0
            mem.push(states[ind, self.action_space.shape[0]:], actions[ind].astype(states.dtype), [mask],
                     next_state[self.action_space.shape[0]:], [reward], None, [task_ind + 1],
                     env_param, states[ind, :self.action_space.shape[0]], [done], [1])
        if self.use_remote:
            logs_ = ray.get([worker.collect_result.remote() for worker in self.workers])
        else:
            logs_ = [worker.collect_result() for worker in self.workers]
        logs = {key: [] for key in logs_[0]}
        for key in logs:
            for item in logs_:
                logs[key] += item[key]
        logs['TotalSteps'] = len(mem)
        return mem, logs

    def get_action(self, state, cur_policy, random, device=torch.device("cpu")):
        with torch.no_grad():
            if random:
                action = self.env.normalization(self.action_space.sample())
            else:
                action = cur_policy.inference_one_step(torch.from_numpy(state[None]).to(device=device,
                                                                         dtype=torch.get_default_dtype()),
                                                       self.deterministic)[0].to(
                        torch.device('cpu')).numpy()
        return action

    def sample1step1env(self, policy, random=False, device=torch.device('cpu'), env_ind=None, render=False, need_info=False):
        if not policy.inference_check_hidden(1):
            policy.inference_init_hidden(1, device)
        cur_policy = policy
        worker = self.workers[0]
        state = worker.get_current_state()
        action = self.get_action(state, cur_policy, random, device)
        if need_info:
            next_state, reward, done, _, task_ind, env_param, current_steps, info = worker.step(action, env_ind, render, need_info=True)
        else:
            next_state, reward, done, _, task_ind, env_param, current_steps = worker.step(action, env_ind, render, need_info=False)

        if done:
            policy.inference_init_hidden(1, device)
        mem = Memory()
        mask = 0.0 if done else 1.0
        if SKIP_MAX_LEN_DONE and done and current_steps >= self.env._max_episode_steps:
            mask = 1.0
        mem.push(state[self.act_dim:], action.astype(state.dtype), [mask],
                 next_state[self.action_space.shape[0]:], [reward], None,
                 [task_ind + 1], env_param, state[:self.act_dim], [done], [1])
        logs = worker.collect_result()
        self.total_steps += 1
        logs['TotalSteps'] = self.total_steps
        if need_info:
            return mem, logs, info
        return mem, logs

    def collect_samples(self, min_batch, policy=None, need_memory=False):
        for i in range(10):
            try:
                mem, logs = self.sample(min_batch, policy)
                break
            except Exception as e:
                print(f'Error occurs while sampling, the error is {e}, tried time: {i}')
        batch = self.extract_from_memory(mem)
        if need_memory:
            return batch, logs, mem
        return batch, logs

    def update_running_state(self, state):
        pass

    @staticmethod
    def extract_from_memory(mem):
        batch = mem.sample()
        state, action, next_state, reward, mask = np.array(batch.state), np.array(batch.action), np.array(
            batch.next_state), np.array(batch.reward), np.array(batch.mask)
        res = {'state': state, 'action': action, 'next_state': next_state, 'reward': reward, 'mask': mask}
        return res

    def make_env_param_dict(self, parameter_name):
        res = {}
        if self.env_tasks is not None:
            for ind, item in enumerate(self.env_tasks):
                res[ind + 1] = item
        res_interprete = {}
        for k, v in res.items():
            if isinstance(v, dict):
                res_interprete[k] = [v[parameter_name][-1]]
            elif isinstance(v, int):
                res_interprete[k] = v
            elif isinstance(v, list):
                res_interprete[k] = math.sqrt(sum([item**2 for item in v]))
            else:
                raise NotImplementedError(f'type({type(v)}) is not implemented.')
        return res_interprete

    def make_env_param_dict_from_params(self, params):
        res_interprete = {}
        for param in params:
            res_ = self.make_env_param_dict(param)
            for k, v in res_.items():
                if k not in res_interprete:
                    res_interprete[k] = v
                else:
                    res_interprete[k] += v

        return res_interprete

if __name__ == '__main__':
    from envs.nonstationary_env import NonstationaryEnv
    env_name = 'Hopper-v2'
    logger = Logger()
    parameter = logger.parameter
    env = NonstationaryEnv(gym.make(env_name), rand_params=parameter.varying_params)

    ray.init()
    worker_num = 1
    env_array = EnvRemoteArray(parameter=parameter, env_name=env_name,
                               worker_num=worker_num, seed=None,
                               policy_type=Policy, env_decoration=NonstationaryEnv,
                               use_true_parameter=False,
                               env_tasks=env.sample_tasks(10), history_len=0, use_remote=False)
    print(parameter.varying_params)
    paramdict = env_array.make_env_param_dict_from_params(parameter.varying_params)
    print(paramdict)
    configs = Policy.make_config_from_param(parameter)
    net = Policy(env.observation_space.shape[0], env.action_space.shape[0], **configs)
    device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)
    import time
    total_step = 0
    replay_buffer = Memory()
    for _ in range(100):
        t0 = time.time()
        batch, logs, mem = env_array.collect_samples(8000, net, need_memory=True)
        logger.add_tabular_data(**logs)
        replay_buffer.append(mem)

        # for i in range(4000):
        #    mem, logs = env_array.sample1step(net, False, device)
        #    replay_buffer.append(mem)
        #    logger.add_tabular_data(**logs)

        total_step += 4000 * worker_num
        # transitions = env_array.extract_from_memory(memory)
        t1 = time.time()
        logger.log_tabular('TimeInterval (s)', t1-t0)
        logger.log_tabular('EnvSteps', total_step)
        logger.log_tabular('ReplayBufferSize', len(replay_buffer))
        logger.dump_tabular()
        # print(logs)



