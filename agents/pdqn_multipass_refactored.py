import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pyhocon.config_tree import ConfigTree
from torch.autograd.variable import Variable

from agents.memory.memory import Memory
from agents.utils.utils import soft_update_target_network, hard_update_target_network
from agents.utils.noise import OrnsteinUhlenbeckActionNoise
from models.models import MultiPassQActor, ParamActor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiPassPDQNAgentRefactored:

    def __init__(self, conf: ConfigTree, env):
        # Config parameters.
        self.epsilon_initial = conf.get('epsilon_initial', 1.0)
        self.epsilon = self.epsilon_initial
        self.epsilon_final = conf.get('epsilon_final', 0.01)
        self.epsilon_steps = conf.get('epsilon_steps', 1000)
        self.batch_size = conf.get('batch_size', 128)
        self.gamma = conf.get('gamma', 0.99)
        self.replay_memory_size = conf.get('replay_memory_size', 10000)
        self.steps_before_learning = conf.get('steps_before_learning', 0)
        self.learning_rate_actor = conf.learning_rate.get('actor', 0.001)
        self.learning_rate_actor_param = conf.learning_rate.get('actor_param', 0.0001)
        self.inverting_gradients = conf.get('inverting_gradients', True)
        self.clip_grad = conf.get('clip_grad', 10)
        self.zero_index_gradients = conf.get('zero_index_gradients', False)
        self.tau_actor = conf.Polyak_averaging.get('tau_actor', 0.01)
        self.tau_actor_param = conf.Polyak_averaging.get('tau_actor_param', 0.001)
        self.OU_noise = conf.get('OU_noise', True)
        self.replay_memory_size = conf.get('replay_memory_size', 10_000)
        self.hidden_layers_actor = conf.actor_model.get('hidden_layer', [128])
        self.hidden_layers_actor_param = conf.actor_param_model.get('hidden_layer', [100])
        # Up to hear it is cleaned.
        device_name = conf.get('device', 'cuda')

        self.num_actions = env.action_space.spaces[0].n
        self.action_parameter_sizes = np.array([env.action_space.spaces[i].shape[0] for i in range(1,self.num_actions+1)])
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max-self.action_min).detach()
        print([env.action_space.spaces[i].high for i in range(1,self.num_actions+1)])
        self.action_parameter_max_numpy = np.concatenate([env.action_space.spaces[i].high for i in range(1, self.num_actions+1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate([env.action_space.spaces[i].low for i in range(1, self.num_actions+1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)

        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)

        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, mu=0., theta=0.15, sigma=0.0001)

        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device)

        print(self.num_actions+self.action_parameter_size)
        self.actor = MultiPassQActor(env.observation_space.spaces[0].shape[0], self.num_actions,
                                     self.action_parameter_sizes, hidden_layers=self.hidden_layers_actor).to(device)
        self.actor_target = MultiPassQActor(env.observation_space.spaces[0].shape[0], self.num_actions,
                                            self.action_parameter_sizes, hidden_layers=self.hidden_layers_actor).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()

        self.actor_param = ParamActor(env.observation_space.spaces[0].shape[0], self.num_actions,
                                      self.action_parameter_size, hidden_layers=self.hidden_layers_actor_param).to(device)
        self.actor_param_target = ParamActor(env.observation_space.spaces[0].shape[0], self.num_actions,
                                             self.action_parameter_size, hidden_layers=self.hidden_layers_actor_param).to(device)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()

        self.loss_func = F.mse_loss  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor) #, betas=(0.95, 0.999))
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param) #, betas=(0.95, 0.999)) #, weight_decay=critic_l2_reg)


        self._step = 0
        self._episode = 0
        self.updates = 0

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        passthrough_layer = self.actor_param.action_parameters_passthrough_layer
        print(initial_weights.shape)
        print(passthrough_layer.weight.data.size())
        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.device)
        if initial_bias is not None:
            print(initial_bias.shape)
            print(passthrough_layer.bias.data.size())
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.device)
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update_target_network(self.actor_param, self.actor_param_target)

    def update_exploration(self):
        self._episode += 1

        ep = self._episode
        if ep < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    ep / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def compute_single_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor_param.forward(state)

            # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
            rnd = np.random.uniform()
            if rnd < self.epsilon:
                chosen_action_id = np.random.choice(self.num_actions)
                if not self.OU_noise:
                    all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy, self.action_parameter_max_numpy))
            else:
                # select maximum action
                Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                chosen_action_id = np.argmax(Q_a)

            # add noise only to parameters of chosen action
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            offset = np.array([self.action_parameter_sizes[i] for i in range(chosen_action_id)], dtype=int).sum()
            if self.OU_noise and self.noise is not None:
                all_action_parameters[offset:offset + self.action_parameter_sizes[chosen_action_id]] += self.noise.sample()[offset:offset + self.action_parameter_sizes[chosen_action_id]]
            formatted_parameters = []

            for action_id in range(self.num_actions):
                low = self.action_parameter_offsets[action_id]
                high = self.action_parameter_offsets[action_id + 1]
                tmp_action_parameters = all_action_parameters[low: high]
                formatted_parameters.append(tmp_action_parameters)

        return chosen_action_id, formatted_parameters

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):
        assert grad.shape[0] == batch_action_indices.shape[0]
        grad = grad.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            ind = torch.zeros(self.action_parameter_size, dtype=torch.long)
            for a in range(self.num_actions):
                ind[self.action_parameter_offsets[a]:self.action_parameter_offsets[a+1]] = a
            # ind_tile = np.tile(ind, (self.batch_size, 1))
            ind_tile = ind.repeat(self.batch_size, 1).to(self.device)
            actual_index = ind_tile != batch_action_indices[:, np.newaxis]
            grad[actual_index] = 0.
        return grad

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '"+str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def learn(self, replay_buffer: Memory):
        self._step += 1
        if self._step >= self.batch_size and self._step >= self.steps_before_learning:
            self._optimize_td_loss(replay_buffer)
            self.updates += 1

    def _optimize_td_loss(self, replay_buffer: Memory):
        if self._step < self.batch_size or self._step < self.steps_before_learning:
            return
        # Sample a batch from replay memory
        states, actions, rewards, next_states, terminals = replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        # ---------------------- optimize Q-network ----------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()

            # Compute the TD error
            target = rewards + (1 - terminals) * self.gamma * Qprime

        # Compute current Q-values using policy network
        q_values = self.actor(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()

        # ---------------------- optimize actor ----------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        Q = self.actor(states, action_params)
        Q_val = Q
        Q_loss = torch.mean(torch.sum(Q_val, 1))
        self.actor.zero_grad()
        Q_loss.backward()
        from copy import deepcopy
        delta_a = deepcopy(action_params.grad.data)
        # step 2
        action_params = self.actor_param(Variable(states))
        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)
        if self.zero_index_gradients:
            delta_a[:] = self._zero_index_gradients(delta_a, batch_action_indices=actions, inplace=True)

        out = -torch.mul(delta_a, action_params)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

        self.actor_param_optimiser.step()

        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)

    def save_models(self, prefix):
        """
        saves the target actor and critic models
        :param prefix: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), prefix + '_actor.pt')
        torch.save(self.actor_param.state_dict(), prefix + '_actor_param.pt')
        print('Models saved successfully')

    def load_models(self, prefix):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param prefix: the count of episodes iterated (used to find the file name)
        :param target: whether to load the target newtwork too (not necessary for evaluation)
        :return:
        """
        # also try load on CPU if no GPU available?
        self.actor.load_state_dict(torch.load(prefix + '_actor.pt', map_location='cpu'))
        self.actor_param.load_state_dict(torch.load(prefix + '_actor_param.pt', map_location='cpu'))
        print('Models loaded successfully')

