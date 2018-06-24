

import pandas as pd
import matplotlib.pyplot as plt

deep_rl_progress = pd.read_csv('data/deep_rl__openai-2018-06-23-17-28-07-455021/progress.csv')
deep_rl_ram_1_hidden_progress = pd.read_csv('data/deep_rl_ram_1_hidden__openai-2018-06-23-19-45-24-528159/progress.csv')
deep_rl_ram_2_hidden_progress = pd.read_csv('data/deep_rl_ram_2_hidden__openai-2018-06-23-20-41-41-153710/progress.csv')
deep_rl_ram_3_hidden_progress = pd.read_csv('data/deep_rl_ram_3_hidden__openai-2018-06-23-21-47-46-256420/progress.csv')

progress = [deep_rl_progress, deep_rl_ram_1_hidden_progress, deep_rl_ram_2_hidden_progress, deep_rl_ram_3_hidden_progress]



mean_100_episode_reward_by_steps = [p.loc[:, ['mean 100 episode reward', 'steps']] for p in progress]

xmax = max([m.loc[:, 'steps'].max() for m in mean_100_episode_reward_by_steps])
xmin = min([m.loc[:, 'steps'].min() for m in mean_100_episode_reward_by_steps])
ymax = max([m.loc[:, 'mean 100 episode reward'].max() for m in mean_100_episode_reward_by_steps])
ymin = min([m.loc[:, 'mean 100 episode reward'].min() for m in mean_100_episode_reward_by_steps])


plt.figure()

for m in mean_100_episode_reward_by_steps:
    plt.plot(m['steps'].tolist(), m['mean 100 episode reward'].tolist())

plt.xlim=(xmin, xmax)
plt.ylim=(ymin, ymax)
plt.xlabel("steps")
plt.ylabel("reward")
plt.legend(['deep_rl', 'deep_rl_ram_1_hidden', 'deep_rl_ram_2_hidden', 'deep_rl_ram_3_hidden'])


mean_reward_exploration = [p.loc[:, ['mean 100 episode reward', '% time spent exploring']].groupby('% time spent exploring')['mean 100 episode reward'].mean().reset_index() for p in progress]

for m in mean_reward_exploration:
    m.columns = ['% time spent exploring', 'mean reward']

xmax = max([m.loc[:, '% time spent exploring'].max() for m in mean_reward_exploration])
xmin = min([m.loc[:, '% time spent exploring'].min() for m in mean_reward_exploration])
ymax = max([m.loc[:, 'mean reward'].max() for m in mean_reward_exploration])
ymin = min([m.loc[:, 'mean reward'].min() for m in mean_reward_exploration])


plt.figure()

for m in mean_reward_exploration:
    plt.plot(m['% time spent exploring'].tolist(), m['mean reward'].tolist())

plt.xlim=(xmin, xmax)
plt.ylim=(ymin, ymax)
plt.xlabel("% time spent exploring")
plt.ylabel("mean reward")
plt.legend(['deep_rl', 'deep_rl_ram_1_hidden', 'deep_rl_ram_2_hidden', 'deep_rl_ram_3_hidden'])

ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])


cummulative_reward_by_time = [p.loc[:, ['steps', 'mean 100 episode reward']] for p in progress]

for r in cummulative_reward_by_time:
    r['cummulative_reward'] = r['mean 100 episode reward'].rolling(min_periods=1, window=len(r)).sum()

xmax = max([r.loc[:, 'steps'].max() for r in cummulative_reward_by_time])
xmin = min([r.loc[:, 'steps'].min() for r in cummulative_reward_by_time])
ymax = max([r.loc[:, 'cummulative_reward'].max() for r in cummulative_reward_by_time])
ymin = min([r.loc[:, 'cummulative_reward'].min() for r in cummulative_reward_by_time])


plt.figure()

for r in cummulative_reward_by_time:
    plt.plot(r['steps'].tolist(), r['cummulative_reward'].tolist())

plt.xlim=(xmin, xmax)
plt.ylim=(ymin, ymax)
plt.xlabel("steps")
plt.ylabel("cummulative reward")
plt.legend(['deep_rl', 'deep_rl_ram_1_hidden', 'deep_rl_ram_2_hidden', 'deep_rl_ram_3_hidden'])


plt.show()