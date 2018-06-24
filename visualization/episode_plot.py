

import pandas as pd
import matplotlib.pyplot as plt
import datetime

deep_rl_episode = pd.read_csv('data/deep_rl__openai-2018-06-23-17-28-07-455021/monitor.csv', header=1)
deep_rl_ram_1_hidden_episode = pd.read_csv('data/deep_rl_ram_1_hidden__openai-2018-06-23-19-45-24-528159/monitor.csv', header=1)
deep_rl_ram_2_hidden_episode = pd.read_csv('data/deep_rl_ram_2_hidden__openai-2018-06-23-20-41-41-153710/monitor.csv', header=1)
deep_rl_ram_3_hidden_episode = pd.read_csv('data/deep_rl_ram_3_hidden__openai-2018-06-23-21-47-46-256420/monitor.csv', header=1)


episode = [deep_rl_episode, deep_rl_ram_1_hidden_episode, deep_rl_ram_2_hidden_episode, deep_rl_ram_3_hidden_episode]


for e in episode:
    e.columns = ['reward', 'length', 'time']
    e.index.name = "episode"


reward_by_episode = [e.loc[:, 'reward'].to_frame() for e in episode]

xmax = max([r.index.max() for r in reward_by_episode])
xmin = min([r.index.min() for r in reward_by_episode])
ymax = max([r.loc[:, 'reward'].max() for r in reward_by_episode])
ymin = min([r.loc[:, 'reward'].min() for r in reward_by_episode])


plt.figure()

for r in reward_by_episode:
    plt.plot(r.index.tolist(), r['reward'].tolist())

plt.xlim=(xmin, xmax)
plt.ylim=(ymin, ymax)
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend(['deep_rl', 'deep_rl_ram_1_hidden', 'deep_rl_ram_2_hidden', 'deep_rl_ram_3_hidden'])


time_by_episode = [e.loc[:, 'time'].to_frame() for e in episode]

xmax = max([t.index.max() for t in time_by_episode])
xmin = min([t.index.min() for t in time_by_episode])
ymax = max([t.loc[:, 'time'].max() for t in time_by_episode])
ymin = min([t.loc[:, 'time'].min() for t in time_by_episode])

plt.figure()

for t in time_by_episode:
    plt.plot(t.index.tolist(), t['time'].tolist())

plt.xlim=(xmin, xmax)
plt.ylim=(ymin, ymax)
plt.xlabel("episode")
plt.ylabel("time")
plt.legend(['deep_rl', 'deep_rl_ram_1_hidden', 'deep_rl_ram_2_hidden', 'deep_rl_ram_3_hidden'])

ax = plt.gca()
ax.set_yticklabels([str(datetime.timedelta(seconds=ytick)) for ytick in ax.get_yticks()])


cummulative_reward_by_time = [e.loc[:, ['time', 'reward']] for e in episode]

for r in cummulative_reward_by_time:
    r['cummulative_reward'] = r['reward'].rolling(min_periods=1, window=len(r)).sum()

xmax = max([r.loc[:, 'time'].max() for r in cummulative_reward_by_time])
xmin = min([r.loc[:, 'time'].min() for r in cummulative_reward_by_time])
ymax = max([r.loc[:, 'cummulative_reward'].max() for r in cummulative_reward_by_time])
ymin = min([r.loc[:, 'cummulative_reward'].min() for r in cummulative_reward_by_time])


plt.figure()

for r in cummulative_reward_by_time:
    plt.plot(r['time'].tolist(), r['cummulative_reward'].tolist())

plt.xlim=(xmin, xmax)
plt.ylim=(ymin, ymax)
plt.xlabel("time")
plt.ylabel("cummulative reward")
plt.legend(['deep_rl', 'deep_rl_ram_1_hidden', 'deep_rl_ram_2_hidden', 'deep_rl_ram_3_hidden'])

ax = plt.gca()
ax.set_xticklabels([str(datetime.timedelta(seconds=xtick)) for xtick in ax.get_xticks()])



plt.show()