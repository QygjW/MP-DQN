import os
import numpy as np
import matplotlib.pyplot as plt
from pyhocon.config_parser import ConfigFactory
from agents.pdqn_multipass_refactored import MultiPassPDQNAgentRefactored
from agents.utils.utils import create_and_wrap_env, evaluate, save_as_fig, modify_paths_according_to_exp
from agents.utils.utils import save_as_csv

# 0 - Reading configs
# 1 - Creating environment and an untrained agent
# 2 - Load the trained models from the checkpoint to the agent
# 3 - Prepare the agent for evaluation process
# 4 - Evaluate the agent's performance under without perturbations
# 5 - Evaluate the agent's behavior under state noise
# 6 - Evaluate the agent's behavior under action noise

# 0.0 - read the configs
conf = ConfigFactory.parse_file('configs/meta_conf.conf')
conf = modify_paths_according_to_exp(conf)
# 0.1 - Getting the evaluation parameters
evaluation_episodes = conf.evaluation.get('evaluation_episodes', 0)
turn_off_exploration = conf.evaluation.get('turn_off_exploration', True)
checkpoint_id = conf.evaluation.get('checkpoint_id', 0)

# 1 - Create env and an agent
env = create_and_wrap_env(conf=conf.env)
agent = MultiPassPDQNAgentRefactored(env=env, conf=conf.agent)

# 2 - Load the model
agent.load_models(os.path.join(conf.saving.save_dir, checkpoint_id))

# 3 - Prepare the agent for evaluation phase
if turn_off_exploration:
    agent.epsilon_final = 0.
    agent.epsilon = 0.
    agent.noise = None

# 4 - Evaluate the agent's performance and create the outputs
print("Evaluating agent over {} episodes".format(evaluation_episodes))
evaluation_returns = evaluate(env, agent, evaluation_episodes)
mean_return = np.mean(evaluation_returns)
std_return = np.std(evaluation_returns)
print('The average performance (total rewards) is R:{0:.6f} (with std:{1:.6f})'.format(mean_return, std_return))

tmp_data = {'experiment_id': conf.experiment_id, 'mean': [mean_return], 'std': [std_return]}
save_as_csv(tmp_data, 'evaluation.csv', conf.evaluation.save_dir)

# 5 - Evaluate the agent's behavior under state noise
N = 10
state_noises_lst = [0.02 * i for i in range(N)]
mean_lst = []
std_lst = []
for state_noise_strength in state_noises_lst:
    evaluation_returns = evaluate(env, agent, evaluation_episodes, state_noise_strength=state_noise_strength)
    mean_lst.append(np.mean(evaluation_returns))
    std_lst.append(np.std(evaluation_returns) / np.sqrt(len(evaluation_returns)))
plt.errorbar(x=state_noises_lst, y=mean_lst, yerr=std_lst)

tmp_data = {'experiment_id': [conf.experiment_id] * N, 'state_noise': state_noises_lst,
            'performance': mean_lst, 'variation': std_lst}
save_as_csv(tmp_data, 'evaluation_state_noise.csv', conf.evaluation.save_dir)
save_as_fig(tmp_data, 'evaluation_state_noise.png', conf.evaluation.save_dir)

# 6 - Evaluate the agent's behavior under action noise
N = 10
action_noises_lst = [0.1 * i for i in range(N)]
mean_lst = []
std_lst = []
for action_noise_strength in action_noises_lst:
    evaluation_returns = evaluate(env, agent, evaluation_episodes, action_noise_strength=action_noise_strength)
    mean_lst.append(np.mean(evaluation_returns))
    std_lst.append(np.std(evaluation_returns) / np.sqrt(len(evaluation_returns)))

tmp_data = {'experiment_id': [conf.experiment_id] * N, 'action_noise': state_noises_lst,
            'performance': mean_lst, 'variation': std_lst}
save_as_csv(tmp_data, 'evaluation_action_noise.csv', conf.evaluation.save_dir)
save_as_fig(tmp_data, 'evaluation_action_noise.png', conf.evaluation.save_dir)
plt.show()

print("Good-bye")
