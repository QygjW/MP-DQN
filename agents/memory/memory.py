"""
Source: https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg.py
"""
import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def clear(self):
        self.start = 0
        self.length = 0
        self.data[:] = 0  # unnecessary, not freeing any memory, could be slow


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, observation_shape, action_shape):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        self.terminals = RingBuffer(limit, shape=(1,))

    def sample(self, batch_size, random_machine=np.random):
        # Draw such that we always have a proceeding element.
        # batch_idxs = random_machine.random_integers(self.nb_entries - 2, size=batch_size)
        batch_idxs = random_machine.random_integers(low=0, high=self.nb_entries-1, size=batch_size)
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        terminals_batch = self.terminals.get_batch(batch_idxs)

        return states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch

    def append(self, state, action, reward, next_state, terminal=False, training=True):
        if not training:
            return

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminals.append(terminal)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.terminals.clear()

    @property
    def nb_entries(self):
        return len(self.states)

