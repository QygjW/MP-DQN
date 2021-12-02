import datetime
import os
from typing import Tuple, Dict, List, Union
import gym.core
import numpy as np

# from agents.mpdqn import MPDQNAgent
from typing import TYPE_CHECKING

from gym.wrappers.monitor import Monitor
from matplotlib import pyplot as plt
from tqdm.asyncio import tqdm
import pandas as pd
import gym_platform
from agents.memory.memory import Memory

if TYPE_CHECKING:
    from agents.pdqn_multipass_refactored import MultiPassPDQNAgentRefactored
from pyhocon.config_tree import ConfigTree

from common.platform_domain import PlatformFlattenedActionWrapper
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper


def save_as_csv(data: Dict, file_name: str, save_dir: str):
    """
    Saves the data into a csv file.

    :param data: a dictionary with the keys for the column names and values to be list of values.
    :param file_name: the name of the csv file.
    :param save_dir: the directory to save the csv file.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_dir, file_name))


def save_as_fig(data: Dict, file_name: str, save_dir: str):
    """
    Saves the data into a png file.

    :param data: a dictionary where the second key is has the x values, the 3rd key has the y, and 4th key the variation
    :param file_name: the name of the csv file.
    :param save_dir: the directory to save the csv file.
    """
    os.makedirs(save_dir, exist_ok=True)
    [_, x_key, y_key, var_key] = list(data.keys())
    plt.close()
    plt.errorbar(x=data[x_key], y=data[y_key], yerr=data[var_key])
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.savefig(os.path.join(save_dir, file_name), format='png')
    plt.close()


def save_as_barplot(data: Union[np.ndarray, List], file_name: str, save_dir: str):
    """
    Saves the data into a png file.

    :param data: a list to be plotted as a bar plot
    :param file_name: the name of the csv file.
    :param save_dir: the directory to save the csv file.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.close()
    plt.bar(range(len(data)), data)
    plt.xlabel('feature id')
    plt.ylabel('importance')
    plt.savefig(os.path.join(save_dir, file_name), format='png')
    plt.close()


def create_replay_buffer(env, conf: ConfigTree) -> Memory:
    """
    Creates an empty replay buffer.

    :param conf: the config which has size of the replay buffer in key size.
    :param env: a gym environment; only the sizes of action and observation space are used.
    :returns: a replay buffer, an instance of Memory.
    """
    replay_buffer_size = conf.get('size', 10_000)
    num_actions = env.action_space.spaces[0].n
    action_parameter_sizes = np.array([env.action_space.spaces[i].shape[0] for i in range(1, num_actions+1)])
    action_parameter_size = int(action_parameter_sizes.sum())
    replay_buffer = Memory(replay_buffer_size, env.observation_space.spaces[0].shape, (1 + action_parameter_size,))
    return replay_buffer


def modify_paths_according_to_exp(conf: ConfigTree) -> ConfigTree:
    """
    Modifies all the path such that every path is a below experiement_id.
    """
    main_res_dir = os.path.join('results', conf.experiment_id)
    conf.saving.__setitem__('save_dir', os.path.join(main_res_dir, conf.saving.save_dir))
    conf.visualization.__setitem__('save_dir', os.path.join(main_res_dir, conf.visualization.save_dir))
    conf.evaluation.__setitem__('save_dir', os.path.join(main_res_dir, conf.evaluation.save_dir))
    conf.explainability.__setitem__('save_dir', os.path.join(main_res_dir, conf.explainability.save_dir))
    return conf


def create_directory_structure(conf: ConfigTree):
    """Creates the necessary directories for saving model and visualizaiton artifacts.

    :param conf: The config with saving and visualization configs
    """
    save_freq = conf.saving.get('save_freq', 0)
    save_dir = conf.saving.get('save_dir', 'checkpoints_' + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    save_frames = conf.visualization.get('save_frames', False)
    visualize = conf.visualization.get('visualize', True)
    render_freq = conf.visualization.get('render_freq', 1)

    if visualize:
        assert render_freq > 0, 'to visualize the render frequency should be larger than 1.'
    if save_frames:
        assert render_freq > 0, 'to save frames the render frequency should be larger than 1.'
        save_dir = conf.visualization.save_dir
        os.makedirs(save_dir, exist_ok=True)


def action_convertor(action_d: int, action_p: np.ndarray) -> Tuple:
    """
    Converts discrete action and parametric action to an action compatible with the env.

    :param action_d: the id of the discrete action
    :param action_p: an array containing the the parameter value for the action_d.
    :returns: action compatible with the environment
    """
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[action_d][:] = action_p
    return action_d, params


def create_and_wrap_env(conf: ConfigTree):
    """
    Creates the environment and applies the wrappers.

    :param conf: the config with the name of the env and boolean flags for different wrappers.
    :returns: an environment
    """
    env_name = conf.get('name', 'Platform-v0')
    seed = conf.get('seed', np.random.randint(low=1, high=np.iinfo(np.int32).max))
    scale_state = conf.get('scale_state', True)
    scale_action = conf.get('scale_action', True)
    flatten_action = conf.get('flatten_action', True)
    monitoring = conf.get('monitoring', False)
    env = gym.make(env_name)
    env.seed(seed)
    if scale_action:
        assert flatten_action, 'to scale the action first they should be flattened.'

    if scale_state:
        env = ScaledStateWrapper(env)
    if flatten_action:
        env = PlatformFlattenedActionWrapper(env)
    if scale_action:
        env = ScaledParameterisedActionWrapper(env)
    # if monitoring:
    #     env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, write_upon_reset=False, force=True)
    return env


def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)

def evaluate(env: gym.core.Env, agent, eval_episodes: int) -> np.ndarray:
    """
    Evaluates the performance of the agent.

    :param env: the environment for the agent.
    :param agent: the agent.
    :param eval_episodes: the number of episodes of evaluations.
    :return: an array of length episodes containing the total reward of the agent.
    """
    return_lst = []
    episode_length_lst = []
    gamma = agent.gamma

    for _ in tqdm(range(eval_episodes)):
        done = False
        state, _ = env.reset()
        episode_length = 0
        return_ = 0.
        while not done:
            state = np.array(state, dtype=np.float32, copy=False)
            action = agent.compute_single_action(state)
            (state, _), reward, done, _ = env.step(action)

            return_ += np.power(gamma, episode_length) * reward
            episode_length += 1

        episode_length_lst.append(episode_length)
        return_lst.append(return_)

    return np.array(return_lst)


def evaluate(env: gym.core.Env, agent, eval_episodes: int, state_noise_strength: float = 0,
             action_noise_strength: float = 0) -> np.ndarray:
    """
    Evaluates the performance of the agent.

    :param env: the environment for the agent.
    :param agent: the agent.
    :param eval_episodes: the number of episodes of evaluations.
    :param state_noise_strength: the stregth of the noise imposed on the state.
    :param action_noise_strength: the strength of the noise imposed on the PARAMETRIC actions.
    :return: an array of length episodes containing the total reward of the agent.
    """
    return_lst = []
    episode_length_lst = []
    gamma = agent.gamma

    for _ in tqdm(range(eval_episodes)):
        done = False
        state, _ = env.reset()
        episode_length = 0
        return_ = 0.
        while not done:
            state = np.array(state, dtype=np.float32, copy=False)
            state[0: 4] += (np.random.rand(len(state[0: 4])) - 0.5) * state_noise_strength
            action = agent.compute_single_action(state)
            action_d = action[0]
            action[1][action_d] += (np.random.rand(len(action[1][action_d])) - 0.5) * action_noise_strength
            (state, _), reward, done, _ = env.step(action)

            return_ += np.power(gamma, episode_length) * reward
            episode_length += 1

        episode_length_lst.append(episode_length)
        return_lst.append(return_)

    return np.array(return_lst)
