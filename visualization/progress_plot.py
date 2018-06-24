

import pandas as pd
import matplotlib.pyplot as plt

deep_rl_progress = pd.read_csv('data/deep_rl__openai-first-execution/progress.csv')
deep_rl_ram_1_hidden_progress = pd.read_csv('data/deep_rl_ram_1_hidden__openai-first-execution/progress.csv')
deep_rl_ram_2_hidden_progress = pd.read_csv('data/deep_rl_ram_2_hidden__openai-first-execution/progress.csv')
deep_rl_ram_3_hidden_progress = pd.read_csv('data/deep_rl_ram_3_hidden__openai-first-execution/progress.csv')
deep_rl_ram_4_hidden_progress = pd.read_csv('data/deep_rl_ram_4_hidden__openai-first-execution/progress.csv')
deep_rl_ram_5_hidden_progress = pd.read_csv('data/deep_rl_ram_5_hidden__openai-first-execution/progress.csv')

progress = [deep_rl_progress, deep_rl_ram_1_hidden_progress, deep_rl_ram_2_hidden_progress, deep_rl_ram_3_hidden_progress, deep_rl_ram_4_hidden_progress, deep_rl_ram_5_hidden_progress]


full_exploitation_timestep = [p.loc[:, ['% time spent exploring', 'steps']][p['% time spent exploring'] <= 1]['steps'].min() for p in progress]

#figure 1
plt.figure()

mean_100_episode_reward_by_steps = [p.loc[:, ['mean 100 episode reward', 'steps']] for p in progress]

xmax = max([m.loc[:, 'steps'].max() for m in mean_100_episode_reward_by_steps])
xmin = min([m.loc[:, 'steps'].min() for m in mean_100_episode_reward_by_steps])
ymax = max([m.loc[:, 'mean 100 episode reward'].max() for m in mean_100_episode_reward_by_steps])
ymin = min([m.loc[:, 'mean 100 episode reward'].min() for m in mean_100_episode_reward_by_steps])

ax = plt.gca()

for i in range(len(mean_100_episode_reward_by_steps)):

    m = mean_100_episode_reward_by_steps[i]
    p = plt.plot(m['steps'].tolist(), m['mean 100 episode reward'].tolist())

    color = p[0].get_color()
    x = full_exploitation_timestep[i]
    ax.vlines(x, ymin, ymax, linestyles='dashdot', colors=color, label="a")
    ax.legend()


plt.xlim=(xmin, xmax)
plt.ylim=(ymin, ymax)
plt.xlabel("steps")
plt.ylabel("reward")
plt.legend(['deep_rl', 'deep_rl_ram_1_hidden', 'deep_rl_ram_2_hidden', 'deep_rl_ram_3_hidden', 'deep_rl_ram_4_hidden', 'deep_rl_ram_5_hidden'])

#figure 2
plt.figure()

cummulative_reward_by_time = [p.loc[:, ['steps', 'mean 100 episode reward']] for p in progress]

for r in cummulative_reward_by_time:
    r['cummulative_reward'] = r['mean 100 episode reward'].rolling(min_periods=1, window=len(r)).sum()

xmax = max([r.loc[:, 'steps'].max() for r in cummulative_reward_by_time])
xmin = min([r.loc[:, 'steps'].min() for r in cummulative_reward_by_time])
ymax = max([r.loc[:, 'cummulative_reward'].max() for r in cummulative_reward_by_time])
ymin = min([r.loc[:, 'cummulative_reward'].min() for r in cummulative_reward_by_time])


ax = plt.gca()

for i in range(len(cummulative_reward_by_time)):

    r = cummulative_reward_by_time[i]
    p = plt.plot(r['steps'].tolist(), r['cummulative_reward'].tolist())

    color = p[0].get_color()
    x = full_exploitation_timestep[i]
    ax.vlines(x, ymin, ymax, linestyles='dashdot', colors=color)


plt.xlim=(xmin, xmax)
plt.ylim=(ymin, ymax)
plt.xlabel("steps")
plt.ylabel("cummulative reward")
plt.legend(['deep_rl', 'deep_rl_ram_1_hidden', 'deep_rl_ram_2_hidden', 'deep_rl_ram_3_hidden', 'deep_rl_ram_4_hidden', 'deep_rl_ram_5_hidden'])

#show figures
plt.show()