import os
import click
import time
import gym
import gym_platform
from gym.wrappers import Monitor

from agents.utils.utils import action_convertor, create_logging_dirs, visualize_and_save_frame, create_and_wrap_env, \
    evaluate, create_replay_buffer
from common import ClickPythonLiteralOption
from common.platform_domain import PlatformFlattenedActionWrapper

import numpy as np

from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper

from pyhocon import ConfigFactory

conf = ConfigFactory.parse_file('conf.conf')

create_logging_dirs(conf.saving)
visualize_and_save_frame(conf.visualization)

env = create_and_wrap_env(conf.env)

from agents.pdqn_multipass_refactored import MultiPassPDQNAgentRefactored
agent = MultiPassPDQNAgentRefactored(env=env, conf=conf.agent)
replay_buffer = create_replay_buffer(env=env, conf=conf.replay_buffer)
total_reward = 0.
returns = []

video_index = 0

for i in range(conf.training.episodes):
    if conf.saving.save_freq > 0 and conf.saving.save_dir and i % conf.saveing.save_freq == 0:
        agent.save_models(os.path.join(conf.saving.save_dir, str(i)))

    done = False
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32, copy=False)

    if conf.visualization.visualize and i % conf.visualization.render_freq == 0:
        env.render()

    episode_reward = 0.

    while not done:
        # Agent takes an action
        act, act_param, all_action_parameters = agent.compute_single_action(state)
        action = action_convertor(act, act_param)
        (next_state, _), reward, done, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32, copy=False)

        replay_buffer.append(state, np.concatenate(([act], all_action_parameters)).ravel(), reward, next_state, done)
        agent.learn(replay_buffer)
        state = next_state
        episode_reward += reward

        if conf.visualization.visualize and i % conf.visualization.render_freq == 0:
            env.render()

    agent.update_exploration()

    if conf.visualization.save_frames and i % conf.visualization.render_freq == 0:
        video_index = env.unwrapped.save_render_states(conf.visualization.save_dir, 'title', video_index)

    returns.append(episode_reward)
    total_reward += episode_reward
    if i % 100 == 0:
        print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))

env.close()
if conf.saving.save_freq > 0 and conf.saving.save_dir:
    agent.save_models(os.path.join(conf.saving.save_dir, str(i)))

returns = env.get_episode_rewards()
print("Ave. return =", sum(returns) / len(returns))
print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

# np.save(os.path.join(dir, conf.experiment_id + "{}".format(str(seed))), returns)
evaluation_episodes = conf.evaluation_episodes
if evaluation_episodes > 0:
    print("Evaluating agent over {} episodes".format(evaluation_episodes))
    agent.epsilon_final = 0.
    agent.epsilon = 0.
    agent.noise = None
    evaluation_returns = evaluate(env, agent, evaluation_episodes)
    print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
    # np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)




print("Good-bye")
