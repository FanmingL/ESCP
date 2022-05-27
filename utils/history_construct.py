import numpy as np
from collections import deque
import copy

class HistoryConstructor:
    """
    state0, action0, state1, action1, ..., state_{max_len}, action_{max_len}, state_{current}
    """
    def __init__(self, max_len, state_dim, action_dim, need_lst_action=False):
        self.max_len = max_len
        self.buffer = deque(maxlen=max_len * 2 + 1)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lst_action = np.zeros((self.action_dim,))
        self.need_lst_action = need_lst_action
        self.reset()

    def __call__(self, current_state):
        if self.max_len == 0:
            if self.need_lst_action:
                obs_dim = len(np.shape(current_state))
                if obs_dim == 1:
                    return np.hstack((self.lst_action.reshape((-1)), current_state))
                else:
                    return np.hstack((self.lst_action.reshape((1, -1)), current_state))
            else:
                return current_state

        self.buffer.append(np.squeeze(current_state))
        return np.hstack(self.buffer)

    def reset(self):
        self.lst_action = np.zeros((self.action_dim, ))
        for i in range(self.max_len):
            self.buffer.append(np.zeros((self.state_dim,)))
            self.buffer.append(np.zeros((self.action_dim,)))

    def update_action(self, action):
        self.lst_action = copy.deepcopy(action)
        if self.max_len == 0:
            return
        self.buffer.append(np.squeeze(action))


if __name__ == '__main__':
    import gym
    env = gym.make('Hopper-v2')
    constructor = HistoryConstructor(4, state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = constructor(env.reset())
    for _ in range(100):
        np.set_printoptions(suppress=True, threshold=int(1e5), linewidth=150, precision=2)
        print('unified state: ', state)
        action = env.action_space.sample()
        next_state_tmp, reward, done, _ = env.step(action)
        print('state: ', next_state_tmp, ', action: ', action)
        constructor.update_action(action)
        next_state = constructor(next_state_tmp)
        state = next_state
