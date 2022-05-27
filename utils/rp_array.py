from utils.replay_memory import MemoryArray


class RPArray:
    def __init__(self, env_num, rnn_slice_length=32, max_trajectory_num=1000, max_traj_step=1050, fix_length=0):
        self.env_num = env_num
        max_trajectory_num_per_env = max_trajectory_num // env_num
        self.max_trajectory_num_per_env = max_trajectory_num_per_env
        self.replay_buffer_array = []
        for i in range(self.env_num):
            self.replay_buffer_array.append(MemoryArray(rnn_slice_length, max_trajectory_num_per_env, max_traj_step, fix_length))

    @property
    def size(self):
        sizes = [item.size for item in self.replay_buffer_array]
        return sum(sizes)

    def sample_transitions(self, batch_size=None):
        pass

    def sample_fix_length_sub_trajs(self, batch_size, fix_length):
        pass

    def sample_trajs(self, batch_size, max_sample_size=None):
        pass

    def mem_push(self, mem):
        pass