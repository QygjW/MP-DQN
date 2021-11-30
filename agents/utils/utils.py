import datetime
import os
from typing import Tuple
import gym.core
import numpy as np

# from agents.mpdqn import MPDQNAgent
from typing import TYPE_CHECKING

from gym.wrappers.monitor import Monitor

if TYPE_CHECKING:
    from agents.pdqn_multipass_refactored import MultiPassPDQNAgentRefactored
from pyhocon.config_tree import ConfigTree

from common.platform_domain import PlatformFlattenedActionWrapper
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper


def create_logging_dirs(conf: ConfigTree):
    """Creates the necessary.

    :param conf: The config with save_dir and save_freq.
    """
    save_freq = conf.get('save_freq', 0)
    save_dir = conf.get('save_dir', 'results_' + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir)
    os.makedirs(save_dir, exist_ok=True)


def visualize_and_save_frame(conf: ConfigTree):
    """
    Visualization setting and saving the frames.

    :param: conf, the configuration for visualization and saving frames.
    """
    save_frames = conf.get('save_frames', False)
    visualize = conf.get('visualize', True)
    render_freq = conf.get('render_freq', 1)

    if visualize:
        assert render_freq > 0, 'to visualize the render frequency should be larger than 1.'
    if save_frames:
        assert render_freq > 0, 'to save frames the render frequency should be larger than 1.'
        save_dir = conf.save_dir
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

    for _ in range(eval_episodes):
        done = False
        state, _ = env.reset()
        episode_length = 0
        return_ = 0.
        while not done:
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.compute_single_action(state)
            action = action_convertor(act, act_param)
            (state, _), reward, done, _ = env.step(action)

            return_ += np.power(gamma, episode_length) * reward
            episode_length += 1

        episode_length_lst.append(episode_length)
        return_lst.append(return_)

    return np.array(return_lst)
