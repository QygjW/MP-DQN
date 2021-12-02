import copy
import os
import numpy as np
from pyhocon.config_parser import ConfigFactory
from tqdm.asyncio import tqdm
from agents.pdqn_multipass_refactored import MultiPassPDQNAgentRefactored
from agents.utils.utils import create_and_wrap_env, save_as_barplot, modify_paths_according_to_exp

# 0.0 - Reading the configs
conf = ConfigFactory.parse_file('configs/meta_conf.conf')
conf = modify_paths_according_to_exp(conf)
turn_off_exploration = conf.explainability.get('turn_off_exploration', True)
checkpoint_id = conf.explainability.get('checkpoint_id', 0)
method = conf.explainability.get('method', 'num_grad')
assert method == 'num_grad', 'methods other than numerical gradients are not implemeneted.'
action_type = conf.explainability.get('action_type', "continues")
assert action_type == "continues", "the discerete action type is not implemented."

epsilon = conf.explainability.get('perturbation_strength', 0.01)
states = conf.explainability.get('states')
save_dir = conf.explainability.get('save_dir', None)
# 0.1 - Instantiations
env = create_and_wrap_env(conf=conf.env)
_, _ = env.reset()
agent = MultiPassPDQNAgentRefactored(env=env, conf=conf.agent)

# 1 - Preparing the agent for explaination!
agent.load_models(os.path.join(conf.saving.save_dir, checkpoint_id))
if turn_off_exploration:
    agent.epsilon_final = 0.
    agent.epsilon = 0.
    agent.noise = None

# 2 - Calculating unscaled gradient for all the samples
states = np.array(states)
N = 1_000
for state_id in range(len(states)):
    state = np.array(states[state_id], dtype=np.float32, copy=True)

    # Calculating the original action
    action = agent.compute_single_action(state)
    action_d_orig = action[0]
    action_p_orig = action[1][action_d_orig]
    grad = np.zeros(len(state))
    for i in tqdm(range(int(N))):
        for state_element in range(len(state)):
            perturbed_state = copy.deepcopy(state)
            noise = (np.random.rand() - 0.5) * epsilon
            perturbed_state[state_element] += noise
            action_pert = agent.compute_single_action(perturbed_state)
            action_p_pert = action_pert[1][action_d_orig]
            grad[state_element] += np.abs(action_p_pert - action_p_orig)

    # normalize
    grad = grad / np.sum(grad)
    if save_dir:
        save_as_barplot(data=grad, file_name='xai_at_state_' + str(state_id) + '.png', save_dir=save_dir)
print("Goodbye")
