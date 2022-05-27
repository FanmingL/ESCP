import copy
import random

import gym
import numpy as np


class GridWorldPlat(gym.Env):
    """
    map:
    ---------------------------------
    | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
    ---------------------------------
    | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 1 |
    ---------------------------------
    | 1 | 2 | 3 | 9 | 3 | 3 | 2 | 1 |
    ---------------------------------
    | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 1 |
    ---------------------------------
    | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
    ---------------------------------
    action0: left/right moving size [-3, -2, -1, 0, 1, 2, 3]
    action1:  up/down moving  size  [-3, -2, -1, 0, 1, 2, 3]
    parameter: moving offset (left/right, up/down)
    {
        'x': random.randint(-3,3),
        'y': random.randint(-3,3)
    }
    """
    def __init__(self, env_flag=(0, 0), append_context=False, offset_size=2, width=11, height=11, moving_size=3):
        super(gym.Env).__init__()
        self.deterministic = True

        self.continuous_action = True
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

        self.observation_space = None
        self._grid_escape_time = 0
        self._grid_max_time = 300
        self.env_flag = env_flag
        self.append_context = append_context

        self.state_space = 2
        self._raw_state_length = self.state_space
        if self.append_context:
            self.state_space += 2
        self.diy_env = True
        self.observation_space = gym.spaces.Box(-1, 1, (self.state_space, ))
        self.width = width
        self.height = height
        self.max_offset = offset_size
        self.max_moving_step_size = moving_size
        self._current_position = (self.width - 1, self.height - 1)
        self.rewards = []
        self._init_reward()
        self.optimal_policy = None

    def _init_reward(self):
        self.rewards = []
        center_x = (self.width - 1) / 2
        center_y = (self.height - 1) / 2
        max_reward = center_x + center_y
        self.rewards = np.zeros((self.height, self.width))
        for x in range(self.width):
            for y in range(self.height):
                self.rewards[y, x] = max_reward - (np.abs(center_x - x) + np.abs(center_y - y))
        # reward has been modified
        # self.rewards[int(center_y), int(center_x)] *= 10
        # self.rewards *= 0.1

    def _init_optimal_policy(self):
        self.optimal_policy = {}
        center_x = (self.width - 1) / 2
        center_y = (self.height - 1) / 2
        for i in range(self.width):
            for j in range(self.height):
                desired_action = [(center_x - i) - self.env_flag[0], (center_y - j) - self.env_flag[1]]
                if desired_action[0] > self.max_moving_step_size:
                    desired_action[0] = self.max_moving_step_size
                elif desired_action[0] < -self.max_moving_step_size:
                    desired_action[0] = -self.max_moving_step_size
                if desired_action[1] > self.max_moving_step_size:
                    desired_action[1] = self.max_moving_step_size
                elif desired_action[1] < -self.max_moving_step_size:
                    desired_action[1] = -self.max_moving_step_size
                self.optimal_policy[(i, j)] = (desired_action[0], desired_action[1])

    @property
    def context(self):
        return [self.env_flag[0] / self.max_offset, self.env_flag[1] / self.max_offset]

    def embed_state(self, state):
        center_x = (self.width - 1) / 2
        center_y = (self.height - 1) / 2

        return [(self._current_position[0] - center_x) / center_x, (self._current_position[1] - center_y) / center_y]

    def get_next_position(self, action):
        x_action = int(action[0] * (self.max_moving_step_size + 1))
        y_action = int(action[1] * (self.max_moving_step_size + 1))
        if x_action > self.max_moving_step_size:
            x_action = self.max_moving_step_size
        if x_action < -self.max_moving_step_size:
            x_action = -self.max_moving_step_size
        if y_action > self.max_moving_step_size:
            y_action = self.max_moving_step_size
        if y_action < -self.max_moving_step_size:
            y_action = -self.max_moving_step_size
        x_action_origin, y_action_origin = x_action, y_action
        x_action += self.env_flag[0]
        y_action += self.env_flag[1]
        possible_x = self._current_position[0] + x_action
        possible_y = self._current_position[1] + y_action
        possible_x = np.clip(possible_x, 0, self.width - 1)
        possible_y = np.clip(possible_y, 0, self.height - 1)
        self._current_position = (int(possible_x), int(possible_y))
        return self._current_position, (x_action_origin, y_action_origin)

    def step(self, action):
        self._grid_escape_time += 1
        done = False
        info = {}

        if self._grid_escape_time >= self._grid_max_time:
            done = True
        info['optimal_action'] = None if self.optimal_policy is None else \
            self.optimal_policy[(self._current_position[0], self._current_position[1])]
        next_position, action_origin = self.get_next_position(action)
        reward = self.rewards[int(next_position[1]), int(next_position[0])]
        next_state_vector = self.embed_state(self._current_position)
        if self.append_context:
            next_state_vector += self.context
        info['next_optimal_action'] = None if self.optimal_policy is None else \
            self.optimal_policy[(self._current_position[0], self._current_position[1])]
        if self.optimal_policy is not None:
            info['action_discrepancy'] = (action_origin[0] - info['optimal_action'][0],
                                          action_origin[1] - info['optimal_action'][1],)
        if (self._current_position[0], self._current_position[1]) == (
            (self.width - 1) / 2,
            (self.height - 1) / 2
        ):
            info['keep_at_target'] = True
        else:
            info['keep_at_target'] = False
        return next_state_vector, reward, done, info

    def reset(self):
        self._grid_escape_time = 0
        self._current_position = (random.randint(0, self.width-1), random.randint(0, self.height-1))
        state = self.embed_state(self._current_position)
        if self.append_context:
            state += self.context
        return state

    def seed(self, seed=None):
        self.action_space.seed(seed)

    def render(self, mode='human'):
        map_rows = []
        nothing_label = 'o'
        have_thing_label = '*'
        for i in range(self.height):
            map_rows.append(nothing_label * self.width + '\n')
        map_rows[self._current_position[1]] = nothing_label*self._current_position[0] + have_thing_label\
                                              + nothing_label*(self.width - self._current_position[0] - 1) + '\n'
        map = ''
        for i in range(self.height):
            map += map_rows[i]
        print(map)



class RandomGridWorldPlat(GridWorldPlat):
    def __init__(self, append_context=False, offset_size=2, width=11, height=11, moving_size=3):
        self.max_offset = offset_size
        self.original_max_offset = self.max_offset
        self.renv_flag = (random.randint(-self.max_offset, self.max_offset),
                          random.randint(-self.max_offset, self.max_offset))
        self.fix_env = None
        super(RandomGridWorldPlat, self).__init__(self.renv_flag, append_context, offset_size, width, height, moving_size)

    def reset(self):
        if self.fix_env is None:
            self.renv_flag = (random.randint(-self.max_offset, self.max_offset),
                              random.randint(-self.max_offset, self.max_offset))
            self.env_flag = self.renv_flag
        else:
            self.renv_flag = self.env_flag = self.fix_env
        self._init_optimal_policy()
        return super(RandomGridWorldPlat, self).reset()

    def set_ood(self, is_ood):
        if is_ood:
            self.max_offset = self.original_max_offset + 1
        else:
            self.max_offset = self.original_max_offset
    def set_fix_env(self, fix_env):
        self.renv_flag = self.env_flag = self.fix_env = fix_env
        self._init_optimal_policy()

    def set_task(self, task):
        self.set_fix_env((task[0], task[0]))

    def sample_tasks(self, n_tasks):
        tasks = []
        task_set = set()
        while len(task_set) < n_tasks:
            task_set.add((random.randint(-self.max_offset, self.max_offset),
                          random.randint(-self.max_offset, self.max_offset)))
        for item in task_set:
            tasks.append([item[0], item[1]])

        return tasks

    @property
    def env_parameter_vector_(self):
        return self.context

    @property
    def env_parameter_length(self):
        return 2

    def plot_reward(self):
        import matplotlib.patches as patches
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('coolwarm')
        def plot_square(ax, x, y, v):
            ax.add_patch(patches.Rectangle(
                (x , y ),
                1.,  # width
                1.,  # height
                facecolor=cm.coolwarm(v),
                # cmap=cmap,
                # c=v,
                edgecolor='black'
            ))
        figure = plt.figure(0, figsize=(4.5, 4/5 * 4.5))
        ax = figure.add_subplot(111)
        for x in range(self.width):
            for y in range(self.height):
                plot_square(ax, x, y, self.rewards[y, x]/np.max(self.rewards))
                if self.rewards[y, x] < 10:
                    plt.text(x+0.4, y+0.4, f'{int(self.rewards[y, x])}')
                else:
                    plt.text(x+0.25, y+0.4, f'{int(self.rewards[y, x])}')

        plt.scatter([-100, -100], [100, 100], c=[0, 1])
        plt.xlim(left=0, right=11)
        plt.ylim(bottom=0, top=11)
        norm = plt.Normalize(vmin=0, vmax=10)
        plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm))
        plt.savefig('grid_demon.pdf')

        plt.show()

from gym.envs.registration import register

register(
id='GridWorldPlat-v2', entry_point=RandomGridWorldPlat
)

def _main():
    import gym
    env = gym.make('GridWorldPlat-v2')
    print('observation space: ', env.observation_space)
    print('action space: ', env.action_space)
    print(hasattr(env, 'rmdm_env_flag'))
    print(env.rewards)
    env.plot_reward()
    exit(0)
    for i in range(10):
        done = False
        state = env.reset()
        print('---' * 18)
        print(env.renv_flag)
        act_space = env.action_space
        action_to_set = None
        while not done:
            action = act_space.sample()
            action = action if action_to_set is None else [action_to_set[0] / 3, action_to_set[1] / 3]
            state, reward, done, info = env.step(action)
            action_to_set = info['next_optimal_action']
            print(info, reward)
            env.render()

if __name__ == '__main__':
    _main()

