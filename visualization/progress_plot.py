

import pandas as pd
import matplotlib.pyplot as plt

progress = pd.read_csv('data/openai-2018-06-23-11-38-48-138281/progress.csv')

mean_100_episode_reward_by_steps = progress.loc[:, ['mean 100 episode reward', 'steps']]

mean_100_episode_reward_by_steps.plot(x='steps', y='mean 100 episode reward')

mean_reward_exploration = progress.loc[:, ['mean 100 episode reward', '% time spent exploring']].groupby('% time spent exploring')['mean 100 episode reward'].mean().reset_index()
mean_reward_exploration.columns = ['% time spent exploring', 'mean reward']

plt.figure()

plt.plot(mean_reward_exploration['% time spent exploring'].tolist(), mean_reward_exploration['mean reward'].tolist())
plt.xlabel("% time spent exploring")
plt.legend(['mean reward'])

ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])

plt.show()