import random

import gym
import numpy as np


class GridWorld(gym.Env):
    def __init__(self, env_flag=2, append_context=False, continuous_action=True):
        super(gym.Env).__init__()
        self.deterministic = True
        # A, B, C, s_0, D
        # ------------------------
        # |    A, B, C   | None  |
        # ------------------------
        # |      s_0     |  D    |
        # ------------------------
        # 0 stay
        # 1 up
        # 2 right
        # 3 left
        # 4 down
        self.continuous_action = continuous_action
        if self.continuous_action:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        else:
            self.action_space = gym.spaces.Discrete(5)

        self.observation_space = None
        self._grid_escape_time = 0
        self._grid_max_time = 1000
        self._current_position = 0
        self.env_flag = env_flag
        self.append_context = append_context
        self.middle_state = [2, 3, 4]
        assert self.env_flag in self.middle_state, '{} is accepted.'.format(self.middle_state)
        self._ind_to_name = {
            0: 's0',
            1: 'D',
            2: 'A',
            3: 'B',
            4: 'C',
            5: 'None'
        }
        self.reward_setting = {
            0: 0,
            1: 1,
            2: 10,
            3: -10,
            4: 0,
            5: 0
        }
        for k in self.reward_setting:
            self.reward_setting[k] *= 0.1

        self.state_space = len(self.reward_setting)
        self._raw_state_length = self.state_space
        if self.append_context:
            self.state_space += len(self.middle_state)
        self.diy_env = True
        self.observation_space = gym.spaces.Box(0, 1, (self.state_space, ))

    @property
    def middle_state_embedding(self):
        v = [0] * len(self.middle_state)
        v[self.env_flag - 2] = 1
        return v

    def make_one_hot(self, state):
        vec = [0] * self._raw_state_length
        vec[state] = 1
        return vec

    def get_next_position_toy(self, action):
        if self._current_position == 0:
            if action == 0:
                # to D
                next_state = 1
            else:
                # to unknown position
                next_state = self.env_flag
        # elif self._current_position == 1:
        #     # keep at D
        #     next_state = 1
        elif self._current_position in self.middle_state + [1]:
            # to s0
            next_state = 0
        else:
            raise NotImplementedError('current position exceeds range!!!')
        return next_state

    def get_next_position(self, action):
        # ------------------------
        # |    A, B, C   | None  |
        # ------------------------
        # |      s_0     |  D    |
        # ------------------------
        # action: 0 stay
        # action: 1 up
        # action: 2 right
        # action: 3 left
        # action: 4 down
        # self._ind_to_name = {
        #             0: 's0',
        #             1: 'D',
        #             2: 'A',
        #             3: 'B',
        #             4: 'C',
        #             5: 'None'
        #         }
        if not self.deterministic:
            if random.random() > 0.5:
                action = action
            else:
                action = random.randint(0, 4)
        if action == 0:
            if self._current_position in [2, 3, 4]:
                return self.env_flag
            return self._current_position
        left_up_map = {
            4: 0,
            # 2: 5
        }
        action_transition_mapping = \
        {
            0: {1: self.env_flag, 2: 1},
            1: {1: 5, 3: 0},
            5: {3: self.env_flag, 4:1},
            2: left_up_map,
            3: left_up_map,
            4: left_up_map
        }
        action_to_state = action_transition_mapping[self._current_position]
        if action in action_to_state:
            return action_to_state[action]
        if self._current_position in [2, 3, 4]:
            return self.env_flag
        return self._current_position

    def step(self, action):
        self._grid_escape_time += 1
        info = {}
        if self.continuous_action:
            action_tmp = (action[0] + 1) / 2
            action_tmp = int(action_tmp * 5)
            if action_tmp >= 5:
                action_tmp = 4
            next_state = self.get_next_position(action_tmp)
        else:
            assert isinstance(action, int), 'action should be int type rather than {}'.format(type(action))
            next_state = self.get_next_position(action)
        done = False # next_state == 1
        if self._grid_escape_time >= self._grid_max_time:
            done = True
        reward = self.reward_setting[next_state]
        info['current_position'] = self._ind_to_name[next_state]
        next_state_vector = self.make_one_hot(next_state)
        self._current_position = next_state
        if self.append_context:
            next_state_vector += self.middle_state_embedding
        return next_state_vector, reward, done, info

    def reset(self):
        self._grid_escape_time = 0
        self._current_position = 0
        state = self.make_one_hot(self._current_position)
        if self.append_context:
            state += self.middle_state_embedding
        return state

    def seed(self, seed=None):
        self.action_space.seed(seed)


class RandomGridWorld(GridWorld):
    def __init__(self, append_context=False):
        self.possible_choice = [2, 3, 4]
        self.renv_flag = random.choice(self.possible_choice)
        self.fix_env = None
        super(RandomGridWorld, self).__init__(self.renv_flag, append_context)

    def reset(self):
        if self.fix_env is None:
            self.renv_flag = random.choice(self.possible_choice)
            self.env_flag = self.renv_flag
        else:
            self.renv_flag = self.env_flag = self.fix_env
        return super(RandomGridWorld, self).reset()

    def set_fix_env(self, fix_env):
        self.renv_flag = self.env_flag = self.fix_env = fix_env

    def set_task(self, task):
        self.set_fix_env(task)

    def sample_tasks(self, n_tasks):
        if n_tasks < len(self.possible_choice):
            tasks = [random.choice(self.possible_choice) for _ in range(n_tasks)]
        else:
            tasks = []
            for i in range(n_tasks):
                tasks.append(self.possible_choice[i % len(self.possible_choice)])

        return tasks

    @property
    def env_parameter_vector_(self):
        return np.array([(self.renv_flag - np.min(self.possible_choice)) / (np.max(self.possible_choice)
                                                                            - np.min(self.possible_choice))])

    @property
    def env_parameter_length(self):
        return 1

from gym.envs.registration import register

register(
id='GridWorldNS-v2', entry_point=RandomGridWorld
)

if __name__ == '__main__':
    import gym
    env = gym.make('GridWorldNS-v2')
    print('observation space: ', env.observation_space)
    print('action space: ', env.action_space)
    print(hasattr(env, 'rmdm_env_flag'))