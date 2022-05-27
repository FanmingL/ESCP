import copy
from collections import namedtuple
import random
from sklearn.neighbors import KernelDensity
import numpy as np
import pickle
import time

# from dppy.finite_dpps import FiniteDPP
# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb
tuplenames = ('state', 'action', 'mask', 'next_state',
                                       'reward', 'next_action', 'task', 'env_param', 'last_action',
                                       'done', 'valid')
Transition = namedtuple('Transition', tuplenames)


class KDE:
    def __init__(self):
        self.min = None
        self.max = None
        self.range = None
        self.dist = None
        self.bandwidth = 0.1

    def norm(self, x):
        if not self.range == 0:
            x = (x - self.min) / self.range
        return x

    def feed(self, x):
        x = np.array(x).reshape(-1, 1)
        self.min = np.min(x)
        self.max = np.max(x)
        self.range = self.max - self.min
        x = self.norm(x)
        self.dist = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian').fit(x)

    def prob(self, x):
        x = np.array(x).reshape(-1, 1)
        x = self.norm(x)
        res = np.exp(self.dist.score_samples(x)).reshape(-1)
        # res = res / (np.max(res) + 0.2)
        return list(res)

class MCMC_MH:
    def __init__(self):
        self.min = None
        self.max = None
        self.range = None
        self.dist = None
        self.bandwidth = 0.1
        self.prob = None
        self.downsample_num = 5
        self.start_num = 200
        self.index = None

    def norm(self, x):
        if not self.range == 0:
            x = (x - self.min) / self.range
        return x

    def feed(self, x):
        x = np.array(x).reshape(-1, 1)
        self.min = np.min(x)
        self.max = np.max(x)
        self.range = self.max - self.min
        x = self.norm(x)
        self.dist = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        self.dist.fit(x)
        self.prob = np.exp(self.dist.score_samples(x)).reshape(-1)
        self.index = [*range(len(self.prob))]

    def update(self, x, prob_x):
        x_star = random.choice(self.index)
        prob_star = self.prob[x_star]
        A = min(1, prob_x / prob_star)
        if random.random() <= A:
            x = x_star
            prob_x = prob_star
        else:
            x = x
            prob_x = prob_x
        return x, prob_x

    def sample(self, batch_size):
        x = random.choice(self.index)
        prob_x = self.prob[x]
        res = []
        for i in range(self.start_num):
            x, prob_x = self.update(x, prob_x)
        for i in range(batch_size * self.downsample_num):
            x, prob_x = self.update(x, prob_x)
            if i % self.downsample_num == 0:
                res.append(x)
        return res
# 
# from dppy.finite_dpps import FiniteDPP
# class DPP:
#     def __init__(self):
#         self.dpp = None
#         self.rank = 0
# 
#     def feed_data(self, x):
#         x = np.array(x).reshape((-1, 1))
#         r = np.max(x) - np.min(x)
#         if r > 0:
#             x = (x - np.min(x)) / r * np.pi
#         PHI = np.hstack([np.cos(x), np.sin(x)]).reshape((-1, 2)).T
#         L = PHI.T@PHI
#         self.dpp = FiniteDPP('likelihood', L_gram_factor=PHI)
# 
#     def sample(self, num):
#         inds = []
#         for i in range(num):
#             inds.extend(self.dpp.sample_exact())
#             if len(inds) >= num:
#                 break
#         self.dpp.flush_samples()
#         return inds
# 
#     def info(self):
#         if self.dpp is not None:
#             self.dpp.info()
# 


class Memory(object):
    def __init__(self):
        self.memory = []
        self.max_size = -1

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None or batch_size > len(self.memory):
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

    def set_max_size(self, max_size):
        self.max_size = max_size

    def limit_size(self):
        if len(self.memory) > self.max_size > 0:
            self.memory[:len(self.memory)-self.max_size] = []

    @property
    def size(self):
        return len(self)


class UniformSample:
    def __init__(self):
        self.sorted_ind = None
        self.x = None
        self.x_min, self.x_max, self.range = None, None, None
        self.epsilon = 1 / 20

    def feed(self, x):
        self.sorted_ind = np.argsort(x)
        self.x_min = x[self.sorted_ind[0]]
        self.x_max = x[self.sorted_ind[-1]]
        self.range = self.x_max - self.x_min
        self.x = x

    def get_ind_from_anchor(self, rands):
        rands = np.sort(rands)
        res_ind = []
        ind_x = 0
        if len(self.x) == 1:
            return [0] * len(rands)
        for item in rands:
            choosed_ind = None
            while True:
                if ind_x >= len(self.sorted_ind):
                    break
                if ind_x == 0:
                    if item <= self.x[self.sorted_ind[ind_x]]:
                        choosed_ind = self.sorted_ind[ind_x]
                        break
                    elif self.x[self.sorted_ind[ind_x]] < item < self.x[self.sorted_ind[ind_x + 1]]:
                        if item - self.x[self.sorted_ind[ind_x]] < self.x[self.sorted_ind[ind_x + 1]] - item:
                            choosed_ind = self.sorted_ind[ind_x]
                        else:
                            choosed_ind = self.sorted_ind[ind_x + 1]
                        break
                elif ind_x == len(self.x) - 1:
                    if item >= self.x[self.sorted_ind[ind_x]]:
                        choosed_ind = self.sorted_ind[ind_x]
                        break
                    elif self.x[self.sorted_ind[ind_x - 1]] < item < self.x[self.sorted_ind[ind_x]]:
                        if item - self.x[self.sorted_ind[ind_x - 1]] < self.x[self.sorted_ind[ind_x]] - item:
                            choosed_ind = self.sorted_ind[ind_x - 1]
                        else:
                            choosed_ind = self.sorted_ind[ind_x]
                        break
                else:
                    if item == self.x[self.sorted_ind[ind_x]]:
                        choosed_ind = self.sorted_ind[ind_x]
                        break
                    elif self.x[self.sorted_ind[ind_x - 1]] < item < self.x[self.sorted_ind[ind_x]]:
                        if item - self.x[self.sorted_ind[ind_x - 1]] < self.x[self.sorted_ind[ind_x]] - item:
                            choosed_ind = self.sorted_ind[ind_x - 1]
                        else:
                            choosed_ind = self.sorted_ind[ind_x]
                        break
                    elif self.x[self.sorted_ind[ind_x]] < item < self.x[self.sorted_ind[ind_x + 1]]:
                        if item - self.x[self.sorted_ind[ind_x]] < self.x[self.sorted_ind[ind_x + 1]] - item:
                            choosed_ind = self.sorted_ind[ind_x]
                        else:
                            choosed_ind = self.sorted_ind[ind_x + 1]
                        break
                ind_x += 1
            assert choosed_ind is not None, 'choosed ind should be assigned value!!!'
            res_ind.append(choosed_ind)
        return res_ind

    def sample(self, batch_size):
        rands = np.random.uniform(self.x_min,
                                  self.x_max,
                                  batch_size)
        return self.get_ind_from_anchor(rands)


class MemoryNp(object):
    def __init__(self, uniform_sample=False, rnn_slice_length=32):
        self.memory = []
        self.memory_buffer = []
        self.rets = []
        self.max_size = -1
        self.transition_count = 0
        self.all_memory_list = []
        self.all_memory_traj_ind = 0
        self.uniform_sample = uniform_sample
        self.sampler_uni = UniformSample()
        self.rnn_slice_length = rnn_slice_length
        self._last_saving_time = 0
        self._last_saving_size = 0

    def get_return(self, traj):
        rews = [item.reward[0] for item in traj]
        return sum(rews)

    def update_memory_list(self, trajs):
        for traj in trajs:
            length = len(traj)
            ret = self.get_return(traj)
            for i in range(length):
                self.all_memory_list.append((self.all_memory_traj_ind, i, ret))
            self.all_memory_traj_ind += 1
            if self.all_memory_list[0][0] > 1e6:
                base = self.all_memory_list[0][0]
                self.all_memory_traj_ind -= base
                for i in range(len(self.all_memory_list)):
                    pre = self.all_memory_list[i]
                    self.all_memory_list[i] = (pre[0] - base, pre[1], pre[2])

    def pop_memory_list(self, trajs):
        length = 0
        for traj in trajs:
            length += len(traj)
        self.all_memory_list[:length] = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

        self.transition_count += 1
        if len(args) == 10 and args[9][0]:
            self.memory_buffer.append(self.memory)
            self.rets.append(self.get_return(self.memory))
            self.update_memory_list([self.memory])
            self.all_memory_traj_ind += 1
            self.memory = []
            self.limit_size()
            if self.uniform_sample:
                self.sampler_uni.feed(self.rets)

        # self.limit_size()

    def sample(self, batch_size=None):
        if batch_size is None or batch_size > len(self.memory):
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def sample_transitions(self, batch_size=None):
        # if batch_size is None or batch_size
        if batch_size is None or batch_size > self.transition_count:
            res = []
            for item in self.memory_buffer:
                res.extend(item)
        else:
            if not self.uniform_sample:
                ind = random.sample(self.all_memory_list, batch_size)
                base = self.all_memory_list[0][0]
                res = [self.memory_buffer[item[0] - base][item[1]] for item in ind]
            else:
                num = 3
                mini_batch_size = batch_size // num
                ind = random.sample(self.all_memory_list, batch_size - mini_batch_size * (num - 1))
                base = self.all_memory_list[0][0]
                res1 = [self.memory_buffer[item[0] - base][item[1]] for item in ind]

                traj_index = self.sampler_uni.sample(mini_batch_size)
                res2 = []
                for ind in traj_index:
                    thing = self.memory_buffer[ind][random.randint(0, len(self.memory_buffer[ind]) - 1)]
                    res2.append(thing)

                # all_memory_list = []
                # traj_index = self.sampler_uni.sample(mini_batch_size)
                # for traj_ind in traj_index:
                #     for i in range(len(self.memory_buffer[traj_ind])):
                #         all_memory_list.append((traj_ind, i))
                # ind = random.sample(all_memory_list, mini_batch_size)
                # res2 = [self.memory_buffer[item[0]][item[1]] for item in ind]

                ind = random.sample(self.all_memory_list[-50 * 1000:], mini_batch_size)
                base = self.all_memory_list[0][0]
                res3 = [self.memory_buffer[item[0] - base][item[1]] for item in ind]

                res = res1 + res2 + res3
        res = Transition(*zip(*res))
        return res

    def make_3dim_buffer(self, trajs, pending_zero_to_back=True, max_len=None):
        # 0-dim trajectory index
        # 1-dim traisitin index
        # 2-dim element
        if max_len is None:
            max_len = self.get_max_len(max([len(traj) for traj in trajs]), self.rnn_slice_length)
        total_data = []         # match 2-dim
        total_elements_num = len(trajs[0][0])
        for i in range(total_elements_num):
            if trajs[0][0][i] is None:
                total_data.append(None)
                continue
            it_datas = []
            for traj in trajs:
                items = [item[i] for item in traj]
                item_np = np.array(items)
                if len(item_np.shape) == 1:
                    item_np = item_np.reshape((-1, 1))
                items_shape = np.shape(item_np)
                if items_shape[0] < max_len:
                    zero_shape = [l for l in items_shape]
                    zero_shape[0] = max_len - items_shape[0]
                    if pending_zero_to_back:
                        item_np = np.vstack((item_np, np.zeros(zero_shape)))
                    else:
                        item_np = np.vstack((np.zeros(zero_shape), item_np))
                it_datas.append(item_np)
            it_datas = np.array(it_datas)
            total_data.append(it_datas)
        return Transition(*total_data)

    def get_max_len(self, max_len, slice_length):
        if max_len % slice_length == 0 and max_len > 0:
            return max_len
        else:
            max_len = (max_len // slice_length + 1) * slice_length
        return max_len

    def pred_max_size(self, trajs):
        max_len = self.get_max_len(max([len(traj) for traj in trajs]), self.rnn_slice_length)
        return max_len * len(trajs)

    def get_maximal_size(self, trajs, max_sample_size):
        max_size = 0
        cur_num = 0
        res = []
        total_size = 0
        for item in trajs:
            if len(item) > max_size:
                max_size = self.get_max_len(len(item), self.rnn_slice_length)
            cur_num += 1
            if cur_num * max_size > max_sample_size:
                cur_num -= 1
                break
            total_size += len(item)
        res = trajs[:cur_num]
        return res, total_size

    def sample_fix_length_sub_trajs(self, batch_size, fix_length):
        ind = random.sample(self.all_memory_list, batch_size)
        base = self.all_memory_list[0][0]
        res = [(item[0] - base, item[1]) for item in ind]
        trajs = []
        for item in res:
            traj_ind = item[0]
            point_ind = item[1]
            traj = self.memory_buffer[traj_ind][max(point_ind-fix_length+1, 0):point_ind+1]

            assert len(traj) <= fix_length, f'expected length: {fix_length}, got {len(traj)}'
            trajs.append(traj)
        res = self.make_3dim_buffer(trajs, pending_zero_to_back=False, max_len=fix_length)
        return res

    def sample_trajs(self, batch_size, max_sample_size=None):
        if False and (batch_size is None or batch_size > self.size):
            trajs = self.memory_buffer
        else:
            if not self.uniform_sample:
                current_points = 0
                trajs = []
                while current_points < batch_size:
                    trajs.append(random.choice(self.memory_buffer))
                    current_points += len(trajs[-1])
            else:
                # not suite for RNN raise an error now
                # raise NotImplementedError('Concurrently, uniform sample is not suited for RNN now!!!!')
                num = 2
                mini_batch_size = batch_size // num
                current_points = 0
                trajs1 = []
                while current_points < batch_size - (mini_batch_size * (num - 1)):
                    trajs1.append(random.choice(self.memory_buffer))
                    current_points += len(trajs1[-1])
                # trajs1 = random.sample(self.memory_buffer, batch_size - mini_batch_size * (num - 1))

                trajs2 = []
                average_len = len(self.all_memory_list) // len(self.memory_buffer)
                expected_traj_num = (mini_batch_size * 3) // average_len
                while current_points < batch_size:
                    traj_index = self.sampler_uni.sample(expected_traj_num)
                    for ind in traj_index:
                        trajs2.append(self.memory_buffer[ind])
                        current_points += len(trajs2[-1])
                        if current_points >= batch_size:
                            break
                trajs = trajs1 + trajs2
        total_size = sum([len(item) for item in trajs])
        if max_sample_size is not None and self.pred_max_size(trajs) > max_sample_size:
            random.shuffle(trajs)
            trajs, total_size = self.get_maximal_size(trajs, max_sample_size)

        res = self.make_3dim_buffer(trajs)
        return res, total_size

    def count_mem_buf(self, mem_buf):
        count = 0
        for item in mem_buf:
            count += len(item)
        return count

    def append(self, new_memory):
        self.memory_buffer += new_memory.memory_buffer
        self.transition_count += self.count_mem_buf(new_memory.memory_buffer)
        self.update_memory_list(new_memory.memory_buffer)
        for traj in new_memory.memory_buffer:
            self.rets.append(self.get_return(traj))
        self.limit_size()
        if self.uniform_sample:
            self.sampler_uni.feed(self.rets)

    def mem_push(self, mem):
        assert len(mem.memory) == 1, "memory should be 1, otherwise, the order will missing!"
        self.transition_count += 1
        self.memory += mem.memory
        if mem.memory[0].done[0]:
            self.memory_buffer.append(self.memory)
            self.rets.append(self.get_return(self.memory))
            self.update_memory_list([self.memory])
            self.memory = []
            self.limit_size()
            if self.uniform_sample:
                self.sampler_uni.feed(self.rets)
        # self.limit_size()

    def mem_push_array(self, mem):
        for item in mem.memory:
            self.transition_count += 1
            self.memory += [item]
            if item.done[0]:
                self.memory_buffer.append(self.memory)
                self.rets.append(self.get_return(self.memory))
                self.update_memory_list([self.memory])
                self.memory = []
                self.limit_size()
                if self.uniform_sample:
                    self.sampler_uni.feed(self.rets)

    def __len__(self):
        return len(self.memory_buffer)

    @property
    def size(self):
        return self.transition_count

    def set_max_size(self, max_size):
        self.max_size = max_size

    def limit_size(self):
        if self.max_size <= 0:
            return
        if self.transition_count < self.max_size:
            return
        delta = self.transition_count - self.max_size
        traj_ind = 0
        for i in range(len(self.memory_buffer)):
            delta -= len(self.memory_buffer[i])
            if delta <= 0:
                traj_ind = i
                break
        self.transition_count -= self.count_mem_buf(self.memory_buffer[:traj_ind+1])
        self.pop_memory_list(self.memory_buffer[:traj_ind+1])
        self.memory_buffer[:traj_ind+1] = []
        self.rets[:traj_ind+1] = []

    def save_to_disk(self, path):
        self._last_saving_time = time.time()
        self._last_saving_size = self.size
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load_from_disk(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for k, v in data.__dict__.items():
            if not k.startswith('_') and not k == 'uniform_sample' and not k == 'rnn_slice_length':
                setattr(self, k, v)

    @staticmethod
    def split_buffer(src, percent):
        traj_num = len(src.memory_buffer)
        dst1 = MemoryNp(src.uniform_sample, src.rnn_slice_length)
        dst1.set_max_size(src.max_size)
        dst1_traj_num = int(traj_num * percent)
        dst2 = MemoryNp(src.uniform_sample, src.rnn_slice_length)
        dst2.set_max_size(src.max_size)
        dst2_traj_num = traj_num - dst1_traj_num

        tmp = MemoryNp(src.uniform_sample, src.rnn_slice_length)
        all_ind = [_ for _ in range(traj_num)]
        dst1_inds = random.sample(all_ind, dst1_traj_num)
        dst1_inds_set = set(dst1_inds)
        dst2_inds = []
        for item in all_ind:
            if item not in dst1_inds_set:
                dst2_inds.append(item)

        tmp.memory_buffer = [src.memory_buffer[ind] for ind in dst1_inds] # [:dst1_traj_num]
        dst1.append(tmp)

        tmp.memory_buffer = [src.memory_buffer[ind] for ind in dst2_inds] # src.memory_buffer[dst1_traj_num:]
        dst2.append(tmp)

        assert len(dst1.memory_buffer) == dst1_traj_num
        assert len(dst2.memory_buffer) == dst2_traj_num
        return dst1, dst2

    def expand_last_action(self):
        for traj in self.memory_buffer:
            last_state = traj[0].state
            for ind, item in enumerate(traj):
                last_action = item.last_action
                last_action = np.concatenate((last_action, last_state), axis=-1)
                tmp = [*item]
                tmp[8] = last_action
                traj[ind] = Transition(*tmp)
                # print(tmp)
                # print(type(item.last_action))
                # item.last_action = last_action
                last_state = traj[ind].state


class MemoryArray(object):
    def __init__(self, rnn_slice_length=32, max_trajectory_num=1000, max_traj_step=1050, fix_length=0):
        self.memory = []
        self.trajectory_length = [0] * max_trajectory_num
        self.available_traj_num = 0
        self.memory_buffer = None
        self.ind_range = None
        self.ptr = 0
        self.max_trajectory_num = max_trajectory_num
        self.max_traj_step = max_traj_step
        self.fix_length = fix_length
        self.transition_buffer = []
        self.transition_count = 0
        self.rnn_slice_length = rnn_slice_length
        self._last_saving_time = 0
        self._last_saving_size = 0
        self.last_sampled_batch = None

    @staticmethod
    def get_max_len(max_len, slice_length):
        if max_len % slice_length == 0 and max_len > 0:
            return max_len
        else:
            max_len = (max_len // slice_length + 1) * slice_length
        return max_len

    def sample_fix_length_sub_trajs(self, batch_size, fix_length):
        list_ind = np.random.randint(0, self.transition_count, (batch_size))
        res = [self.transition_buffer[ind] for ind in list_ind]
        if self.last_sampled_batch is None or not self.last_sampled_batch.shape[0] == batch_size or not self.last_sampled_batch.shape[1] == fix_length:
            trajs = [self.memory_buffer[traj_ind, point_ind+1- fix_length:point_ind+1]for traj_ind, point_ind in res]
            trajs = np.array(trajs, copy=True)
            self.last_sampled_batch = trajs
        else:
            for ind, (traj_ind, point_ind) in enumerate(res):
                self.last_sampled_batch[ind, :, :] = self.memory_buffer[traj_ind,
                                                        point_ind + 1 - fix_length: point_ind + 1, :]

        res = self.array_to_transition(self.last_sampled_batch)
        return res

    def sample_trajs(self, batch_size, max_sample_size=None):
        mean_traj_len = self.transition_count / self.available_traj_num
        desired_traj_num = max(int(batch_size / mean_traj_len), 1)
        if max_sample_size is not None:
            max_traj_num = max_sample_size // self.max_traj_step
            desired_traj_num = min(desired_traj_num, max_traj_num)
        traj_inds = np.random.randint(0, self.available_traj_num, (int(desired_traj_num)))
        trajs = self.memory_buffer[traj_inds]
        traj_len = [self.trajectory_length[ind] for ind in traj_inds]
        max_traj_len = max(traj_len)
        max_traj_len = self.get_max_len(max_traj_len, self.rnn_slice_length)
        trajs = trajs[:, :max_traj_len, :]
        total_size = sum(traj_len)

        return self.array_to_transition(trajs), total_size

    def transition_to_array(self, transition):
        res = []
        for item in transition:
            if isinstance(item, np.ndarray):
                res.append(item.reshape((1, -1)))
            elif isinstance(item, list):
                res.append(np.array(item).reshape((1, -1)))
            elif item is None:
                pass
            else:
                raise NotImplementedError('not implement for type of {}'.format(type(item)))
        res = np.hstack(res)
        assert res.shape[-1] == self.memory_buffer.shape[-1], 'data_size: {}, buffer_size: {}'.format(res.shape, self.memory_buffer.shape)
        return np.hstack(res)

    def array_to_transition(self, data):
        data_list = []
        for item in self.ind_range:
            if len(item) > 0:
                start = item[0]
                end = item[-1] + 1
                data_list.append(data[..., start:end])
            else:
                data_list.append(None)
        res = Transition(*data_list)
        return res

    def complete_traj(self, memory):
        if self.memory_buffer is None:
            print('init replay buffer')
            start_dim = 0
            self.ind_range = []
            self.trajectory_length = [0] * self.max_trajectory_num
            end_dim = 0
            for item in memory[0]:
                dim = 0
                if 'ndarray' in str(type(item)):
                    dim = item.shape[-1]
                elif isinstance(item, list):
                    dim = len(item)
                elif item is None:
                    dim = 0
                end_dim = start_dim + dim
                self.ind_range.append(list(range(start_dim, end_dim)))
                start_dim = end_dim
            for name, ind_range in zip(tuplenames, self.ind_range):
                print(f'name: {name}, ind: {ind_range}')
            self.memory_buffer = np.zeros((self.max_trajectory_num, self.max_traj_step + self.fix_length, end_dim))
        for ind, transition in enumerate(memory):
            self.memory_buffer[self.ptr, ind + self.fix_length, :] = self.transition_to_array(transition)
            self.transition_buffer.append((self.ptr, ind + self.fix_length))
        self.transition_count -= self.trajectory_length[self.ptr]
        if self.trajectory_length[self.ptr] > 0:
            self.transition_buffer[:self.trajectory_length[self.ptr]] = []
        self.trajectory_length[self.ptr] = len(memory)

        self.ptr += 1
        self.available_traj_num = max(self.available_traj_num, self.ptr)
        self.transition_count += len(memory)
        if self.ptr >= self.max_trajectory_num:
            self.ptr = 0

    def remake_transition_buffer(self):
        self.transition_buffer = []
        for ind, item in enumerate(self.trajectory_length):
            for i in range(item):
                self.transition_buffer.append((ind, i + self.fix_length))

    def mem_push_array(self, mem):
        for item in mem.memory:
            self.memory += [item]
            if item.done[0]:
                self.complete_traj(self.memory)
                self.memory = []

    def mem_push(self, mem):
        self.mem_push_array(mem)

    def __len__(self):
        return self.available_traj_num

    @property
    def size(self):
        return self.transition_count

    def save_to_disk(self, path):
        self._last_saving_time = time.time()
        self._last_saving_size = self.size
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=4)

    def load_from_disk(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for k, v in data.__dict__.items():
            if not k.startswith('_') and not k == 'uniform_sample' and not k == 'rnn_slice_length':
                setattr(self, k, v)

    def test_speed(self, fix_length=32):
        self.remake_transition_buffer()
        # self.available_traj_num = 100
        for i in range(self.available_traj_num):
            self.trajectory_length[i] = 800
        for _ in range(100):
            if fix_length > 0:
                self.sample_fix_length_sub_trajs(8000, fix_length)
            else:
                self.sample_trajs(8000, 50000)

    def sample_transitions(self, batch_size=None):
        if batch_size is not None:
            list_ind = np.random.randint(0, self.transition_count, (batch_size))
        else:
            list_ind = list(range(self.transition_count))
        res = [self.transition_buffer[ind] for ind in list_ind]
        trajs = [self.memory_buffer[traj_ind, point_ind] for traj_ind, point_ind
                 in res]
        trajs = np.array(trajs, copy=True)
        res = self.array_to_transition(trajs)
        return res

    def stack_zeros(self, rnn_fix_length):
        self.fix_length = rnn_fix_length
        self.memory_buffer = np.concatenate((np.zeros((self.max_trajectory_num, self.fix_length,
                                                       self.memory_buffer.shape[-1])), self.memory_buffer), axis=1)
        self.remake_transition_buffer()

    def split_buffer(self, ratio):
        a_buffer = copy.deepcopy(self)

        b_buffer = copy.deepcopy(self)

        a_length = int(self.available_traj_num * ratio)
        b_length = int(self.available_traj_num - a_length)
        permutation_ind = np.random.permutation(self.available_traj_num)

        a_ind = permutation_ind[:a_length]
        b_ind = permutation_ind[a_length:]

        a_buffer.trajectory_length = [0] * self.max_trajectory_num
        b_buffer.trajectory_length = [0] * self.max_trajectory_num

        a_buffer.trajectory_length[:a_length] = [self.trajectory_length[item] for item in a_ind]
        b_buffer.trajectory_length[:b_length] = [self.trajectory_length[item] for item in b_ind]
        # print(a_buffer.trajectory_length, [item for item in a_ind])
        # print(b_buffer.trajectory_length, [item for item in b_ind])
        #
        a_buffer.available_traj_num = a_length
        b_buffer.available_traj_num = b_length

        a_buffer.memory_buffer[:a_length] = self.memory_buffer[a_ind]
        b_buffer.memory_buffer[:b_length] = self.memory_buffer[b_ind]

        a_buffer.remake_transition_buffer()
        b_buffer.remake_transition_buffer()
        # print(a_length, a_buffer.available_traj_num, self.max_trajectory_num, a_buffer.trajectory_length)
        # print(b_length, b_buffer.available_traj_num, self.max_trajectory_num, b_buffer.trajectory_length)
        # print(a_buffer.trajectory_length,
        #       b_buffer.trajectory_length)
        a_buffer.transition_count = sum(a_buffer.trajectory_length)
        b_buffer.transition_count = sum(b_buffer.trajectory_length)

        return a_buffer, b_buffer


if __name__ == '__main__':
    import cProfile as profile
    fix_length = 32
    memory = MemoryArray(16, 1000, 1050, fix_length=fix_length)
    memory.load_from_disk('/Users/fanmingluo/Code/py/modified_policy_adapatation/log_file/replay_buffer.pkl')
    print(len(memory), memory.size)
    profile.runctx('memory.test_speed({})'.format(fix_length), globals(), locals(), sort=2, )
