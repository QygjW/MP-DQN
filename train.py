import os
import numpy as np
from pyhocon import ConfigFactory
from tqdm.asyncio import tqdm

from agents.utils.utils import create_directory_structure, create_and_wrap_env, create_replay_buffer, \
    modify_paths_according_to_exp, save_as_fig, save_as_csv
from agents.pdqn_multipass_refactored import MultiPassPDQNAgentRefactored as MPDQN

conf = ConfigFactory.parse_file('configs/meta_conf.conf')
conf = modify_paths_according_to_exp(conf)
create_directory_structure(conf=conf)
env = create_and_wrap_env(conf=conf.env)
agent = MPDQN(env=env, conf=conf.agent)
replay_buffer = create_replay_buffer(env=env, conf=conf.replay_buffer)

total_reward = 0.
returns = []

video_index = 0

for i in tqdm(range(conf.agent.training_episodes)):
    if conf.saving.save_freq > 0 and conf.saving.save_dir and i % conf.saving.save_freq == 0:
        agent.save_models(os.path.join(conf.saving.save_dir, str(i)))

    done = False
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32, copy=False)

    if conf.visualization.visualize and i % conf.visualization.render_freq == 0:
        env.render()

    episode_reward = 0.

    while not done:
        action = agent.compute_single_action(state)
        (next_state, _), reward, done, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32, copy=False)
        replay_buffer.append(state, action, reward, next_state, done)
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
        print('{0:5s} Total average reward:{1:.4f} Avergae 100 last rewards:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))

env.close()

if conf.saving.save_freq > 0 and conf.saving.save_dir:
    agent.save_models(os.path.join(conf.saving.save_dir, str(i)))

returns = np.array(returns)
print("Ave. return =", sum(returns) / len(returns))
print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)
training_progress = {'0': None, 'iteration': list(range(len(returns))), 'performance:': returns, 'var': None}
save_as_fig(training_progress, 'training_curve.png', conf.evaluation.save_dir)
save_as_csv(training_progress, 'training_curve.csv', conf.evaluation.save_dir)
print("Good-bye")
