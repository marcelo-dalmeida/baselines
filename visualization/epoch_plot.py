

import pandas as pd
import matplotlib.pyplot as plt
import datetime

epoch = pd.read_csv('data/openai-2018-06-23-11-38-48-138281/monitor.csv', header=1)
epoch.columns = ['reward', 'length', 'time']
epoch.index.name = "epoch"

print(epoch)

epoch.loc[:, 'reward'].to_frame().plot(y='reward')

plt.figure()

plt.plot(epoch.index.tolist(), epoch['time'].tolist())
plt.xlabel("epoch")
plt.legend(['time'])

ax = plt.gca()
ax.set_yticklabels([str(datetime.timedelta(seconds=ytick)) for ytick in ax.get_yticks()])


plt.figure()

plt.plot(epoch['time'].tolist(), epoch['reward'].rolling(min_periods=1, window=len(epoch)).sum().tolist())
plt.xlabel("time")
plt.legend(['reward'])

ax = plt.gca()
ax.set_xticklabels([str(datetime.timedelta(seconds=xtick)) for xtick in ax.get_xticks()])



plt.show()