import os
import click
import time
import gym
import gym_platform
from gym.wrappers import Monitor

from agents.utils.utils import action_convertor, create_logging_dirs, visualize_and_save_frame, create_and_wrap_env, evaluate
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

print(agent)
max_steps = 250
total_reward = 0.
returns = []
start_time = time.time()
video_index = 0

for i in range(conf.training.episodes):
    if conf.saving.save_freq > 0 and conf.saving.save_dir and i % conf.saveing.save_freq == 0:
        agent.save_models(os.path.join(conf.saving.save_dir, str(i)))
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32, copy=False)
    if conf.visualization.visualize and i % conf.visualization.render_freq == 0:
        env.render()

    act, act_param, all_action_parameters = agent.act(state)
    action = action_convertor(act, act_param)

    episode_reward = 0.
    agent.start_episode()
    for j in range(max_steps):

        ret = env.step(action)
        (next_state, steps), reward, terminal, _ = ret
        next_state = np.array(next_state, dtype=np.float32, copy=False)

        next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
        next_action = action_convertor(next_act, next_act_param)
        agent.step(state, (act, all_action_parameters), reward, next_state,
                   (next_act, next_all_action_parameters), terminal, steps)
        act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
        action = next_action
        state = next_state

        episode_reward += reward
        if conf.visualization.visualize and i % conf.visualization.render_freq == 0:
            env.render()

        if terminal:
            break
    agent.end_episode()

    if conf.visualization.save_frames and i % conf.visualization.render_freq == 0:
        video_index = env.unwrapped.save_render_states(conf.visualization.save_dir, 'title', video_index)

    returns.append(episode_reward)
    total_reward += episode_reward
    if i % 100 == 0:
        print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))
end_time = time.time()
print("Took %.2f seconds" % (end_time - start_time))
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
