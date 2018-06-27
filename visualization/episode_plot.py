

import pandas as pd
import matplotlib.pyplot as plt
import datetime

def plot(episode, legend):
    for e in episode:
        e.columns = ['reward', 'length', 'time']
        e.index.name = "episode"


    #figure 1
    plt.figure()

    reward_by_episode = [e.loc[:, 'reward'].to_frame() for e in episode]

    xmax = max([r.index.max() for r in reward_by_episode])
    xmin = min([r.index.min() for r in reward_by_episode])
    ymax = max([r.loc[:, 'reward'].max() for r in reward_by_episode])
    ymin = min([r.loc[:, 'reward'].min() for r in reward_by_episode])


    for r in reward_by_episode:
        plt.plot(r.index.tolist(), r['reward'].tolist())

    plt.xlim=(xmin, xmax)
    plt.ylim=(ymin, ymax)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend(legend)


    #figure 2
    plt.figure()

    cummulative_reward_by_time = [e.loc[:, ['time', 'reward']] for e in episode]

    for r in cummulative_reward_by_time:
        r['cummulative reward'] = r['reward'].rolling(min_periods=1, window=len(r)).sum()

    xmax = max([r.loc[:, 'time'].max() for r in cummulative_reward_by_time])
    xmin = min([r.loc[:, 'time'].min() for r in cummulative_reward_by_time])
    ymax = max([r.loc[:, 'cummulative reward'].max() for r in cummulative_reward_by_time])
    ymin = min([r.loc[:, 'cummulative reward'].min() for r in cummulative_reward_by_time])


    for r in cummulative_reward_by_time:
        plt.plot(r['time'].tolist(), r['cummulative reward'].tolist())

    plt.xlim=(xmin, xmax)
    plt.ylim=(ymin, ymax)
    plt.xlabel("time")
    plt.ylabel("cummulative reward")
    plt.legend(legend)

    ax = plt.gca()
    ax.set_xticklabels([str(datetime.timedelta(seconds=xtick)) for xtick in ax.get_xticks()])


    #figure 3
    plt.figure()

    episode_by_time = [e.loc[:, 'time'].to_frame() for e in episode]

    xmax = max([t.loc[:, 'time'].max() for t in episode_by_time])
    xmin = min([t.loc[:, 'time'].min() for t in episode_by_time])
    ymax = max([t.index.max() for t in episode_by_time])
    ymin = min([t.index.min() for t in episode_by_time])


    for t in episode_by_time:
        plt.plot(t['time'].tolist(), t.index.tolist())

    plt.xlim=(xmin, xmax)
    plt.ylim=(ymin, ymax)
    plt.xlabel("time")
    plt.ylabel("episode")
    plt.legend(legend)

    ax = plt.gca()
    ax.set_xticklabels([str(datetime.timedelta(seconds=xtick)) for xtick in ax.get_xticks()])


    #show figures
    plt.show()


def plot_laptop_versus_desktop():
    deep_rl_episode_desktop = pd.read_csv('data/deep_rl__openai-first-execution/monitor.csv', header=1)
    deep_rl_episode_laptop = pd.read_csv('data/laptop-deep_rl__openai-execution/monitor.csv', header=1)

    episode = [deep_rl_episode_desktop, deep_rl_episode_laptop]
    legend = ['deep_rl @ desktop', 'deep_rl @ laptop']

    plot(episode, legend)

def plot_all_techniques_comparison():
    deep_rl_episode = pd.read_csv('data/deep_rl__openai-second-execution/monitor.csv', header=1)
    deep_rl_ram_1_hidden_episode = pd.read_csv('data/deep_rl_ram_1_hidden__openai-second-execution/monitor.csv',
                                               header=1)
    deep_rl_ram_2_hidden_episode = pd.read_csv('data/deep_rl_ram_2_hidden__openai-second-execution/monitor.csv',
                                               header=1)
    deep_rl_ram_3_hidden_episode = pd.read_csv('data/deep_rl_ram_3_hidden__openai-second-execution/monitor.csv',
                                               header=1)
    deep_rl_ram_4_hidden_episode = pd.read_csv('data/deep_rl_ram_4_hidden__openai-first-execution/monitor.csv',
                                               header=1)
    deep_rl_ram_5_hidden_episode = pd.read_csv('data/deep_rl_ram_5_hidden__openai-third-execution/monitor.csv',
                                               header=1)

    episode = [deep_rl_episode, deep_rl_ram_1_hidden_episode, deep_rl_ram_2_hidden_episode,
               deep_rl_ram_3_hidden_episode, deep_rl_ram_4_hidden_episode, deep_rl_ram_5_hidden_episode]
    legend = ['deep_rl', 'deep_rl_ram_1_hidden', 'deep_rl_ram_2_hidden', 'deep_rl_ram_3_hidden', 'deep_rl_ram_4_hidden',
              'deep_rl_ram_5_hidden']

    plot(episode, legend)

def plot_same_architecture_comparison():

    deep_rl_episode_first = pd.read_csv('data/deep_rl__openai-first-execution/monitor.csv', header=1)
    deep_rl_episode_second = pd.read_csv('data/deep_rl__openai-second-execution/monitor.csv', header=1)
    deep_rl_episode_third = pd.read_csv('data/deep_rl__openai-third-execution/monitor.csv', header=1)

    episode = [deep_rl_episode_first, deep_rl_episode_second, deep_rl_episode_third]
    legend = ['deep_rl [first execution]', 'deep_rl [second execution]', 'deep_rl [third execution]']
    plot(episode, legend)


    deep_rl_ram_1_hidden_episode_first = pd.read_csv('data/deep_rl_ram_1_hidden__openai-first-execution/monitor.csv', header=1)
    deep_rl_ram_1_hidden_episode_second = pd.read_csv('data/deep_rl_ram_1_hidden__openai-second-execution/monitor.csv', header=1)
    deep_rl_ram_1_hidden_episode_third = pd.read_csv('data/deep_rl_ram_1_hidden__openai-third-execution/monitor.csv', header=1)

    episode = [deep_rl_ram_1_hidden_episode_first, deep_rl_ram_1_hidden_episode_second, deep_rl_ram_1_hidden_episode_third]
    legend = ['deep_rl_ram_1_hidden [first execution]', 'deep_rl_ram_1_hidden [second execution]', 'deep_rl_ram_1_hidden [third execution]']
    plot(episode, legend)


    deep_rl_ram_2_hidden_episode_first = pd.read_csv('data/deep_rl_ram_2_hidden__openai-first-execution/monitor.csv', header=1)
    deep_rl_ram_2_hidden_episode_second = pd.read_csv('data/deep_rl_ram_2_hidden__openai-second-execution/monitor.csv', header=1)
    deep_rl_ram_2_hidden_episode_third = pd.read_csv('data/deep_rl_ram_2_hidden__openai-third-execution/monitor.csv', header=1)

    episode = [deep_rl_ram_2_hidden_episode_first, deep_rl_ram_2_hidden_episode_second, deep_rl_ram_2_hidden_episode_third]
    legend = ['deep_rl_ram_2_hidden [first execution]', 'deep_rl_ram_2_hidden [second execution]', 'deep_rl_ram_2_hidden [third execution]']
    plot(episode, legend)


    deep_rl_ram_3_hidden_episode_first = pd.read_csv('data/deep_rl_ram_3_hidden__openai-first-execution/monitor.csv', header=1)
    deep_rl_ram_3_hidden_episode_second = pd.read_csv('data/deep_rl_ram_3_hidden__openai-second-execution/monitor.csv', header=1)
    deep_rl_ram_3_hidden_episode_third = pd.read_csv('data/deep_rl_ram_3_hidden__openai-third-execution/monitor.csv', header=1)

    episode = [deep_rl_ram_3_hidden_episode_first, deep_rl_ram_3_hidden_episode_second, deep_rl_ram_3_hidden_episode_third]
    legend = ['deep_rl_ram_3_hidden [first execution]', 'deep_rl_ram_3_hidden [second execution]', 'deep_rl_ram_3_hidden [third execution]']
    plot(episode, legend)


    deep_rl_ram_4_hidden_episode_first = pd.read_csv('data/deep_rl_ram_4_hidden__openai-first-execution/monitor.csv', header=1)
    deep_rl_ram_4_hidden_episode_second = pd.read_csv('data/deep_rl_ram_4_hidden__openai-second-execution/monitor.csv', header=1)
    deep_rl_ram_4_hidden_episode_third = pd.read_csv('data/deep_rl_ram_4_hidden__openai-third-execution/monitor.csv', header=1)

    episode = [deep_rl_ram_4_hidden_episode_first, deep_rl_ram_4_hidden_episode_second, deep_rl_ram_4_hidden_episode_third]
    legend = ['deep_rl_ram_4_hidden [first execution]', 'deep_rl_ram_4_hidden [second execution]', 'deep_rl_ram_4_hidden [third execution]']
    plot(episode, legend)


    deep_rl_ram_5_hidden_episode_first = pd.read_csv('data/deep_rl_ram_5_hidden__openai-first-execution/monitor.csv', header=1)
    deep_rl_ram_5_hidden_episode_second = pd.read_csv('data/deep_rl_ram_5_hidden__openai-second-execution/monitor.csv', header=1)
    deep_rl_ram_5_hidden_episode_third = pd.read_csv('data/deep_rl_ram_5_hidden__openai-third-execution/monitor.csv', header=1)

    episode = [deep_rl_ram_5_hidden_episode_first, deep_rl_ram_5_hidden_episode_second, deep_rl_ram_5_hidden_episode_third]
    legend = ['deep_rl_ram_5_hidden [first execution]', 'deep_rl_ram_5_hidden [second execution]', 'deep_rl_ram_5_hidden [third execution]']
    plot(episode, legend)



plot_laptop_versus_desktop()

plot_all_techniques_comparison()

plot_same_architecture_comparison()