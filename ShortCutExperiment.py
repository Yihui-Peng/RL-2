# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.

import io
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from ShortCutAgents import QLearningAgent
from ShortCutAgents import SARSAAgent
from ShortCutEnvironment import WindyShortcutEnvironment
from ShortCutAgents import ExpectedSARSAAgent
from ShortCutAgents import nStepSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment

# adding **agent_kwargs to make the n-steps work
def run_repetitions(env_class, agent_class, n_rep, n_episodes, alpha=0.1, epsilon=0.1, **agent_kwargs):
    all_returns = []
    for _ in range(n_rep):
        env = env_class()
        agent = agent_class(
            n_actions=env.action_size(),
            n_states=env.state_size(),
            epsilon=epsilon,
            alpha=alpha,
            **agent_kwargs
        )
        returns = agent.train(env, n_episodes)
        all_returns.append(returns)
    return np.array(all_returns)


def plot_learning_curves(data, title, xlabel='Episode', ylabel='Cumulative Reward', labels=None, smooth_window=10):
    plt.figure()
    for i, curve in enumerate(data):
        if smooth_window > 1:
            smoothed = np.convolve(curve, np.ones(smooth_window)/smooth_window, mode='valid')
            plt.plot(smoothed, label=labels[i] if labels else None)
        else:
            plt.plot(curve, label=labels[i] if labels else None)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels:
        plt.legend()
    plt.show()


def filter_ansi_escape(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def fix_arrows_near_goal(grid):
    goal_y, goal_x = 8, 8
    for dy, dx, arrow in [(-1, 0, '↓'), (1, 0, '↑'), (0, -1, '→'), (0, 1, '←')]:
        ny, nx = goal_y + dy, goal_x + dx
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
            if grid[ny, nx] not in ['C', 'G']:
                grid[ny, nx] = arrow
    return grid


def capture_render_greedy(env, Q):
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    env.render_greedy(Q)
    sys.stdout = old_stdout
    cleaned_text = filter_ansi_escape(buffer.getvalue())
    grid_lines = [line.strip().split() for line in cleaned_text.split('\n') if line.strip()]
    grid = np.array(grid_lines)
    grid = fix_arrows_near_goal(grid)
    return grid


def plot_greedy_path(grid, title, filename):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', lw=1)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j - 0.5, color='black', lw=1)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            cell = grid[y, x]
            color = 'black'
            if cell in ['↑', '↓', '←', '→']:
                color = 'blue'
            elif cell == 'C':
                color = 'red'
            elif cell == 'G':
                color = 'green'
            elif cell == '0':
                cell = ''
            else:
                color = 'black'
            ax.text(x, y, cell, ha='center', va='center', color=color, fontsize=12)
    ax.text(2, 2, 'S', ha='center', va='center', color='blue', fontsize=14)
    ax.text(9, 2, 'S', ha='center', va='center', color='blue', fontsize=14)
    ax.text(8, 8, 'G', ha='center', va='center', color='green', fontsize=14)
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def run_single_and_render(agent_class, name, env_class=ShortcutEnvironment, n_episodes=10000, alpha=0.5, epsilon=0.1, gamma=0.99, **agent_kwargs):
    print(f"\n==== {name} ====")
    env = env_class()
    agent = agent_class(env.action_size(), env.state_size(), alpha=alpha, epsilon=epsilon, gamma=gamma, **agent_kwargs)
    agent.train(env, n_episodes)
    env.render_greedy(agent.Q)


# Run all algorithms in normal environment
run_single_and_render(QLearningAgent, "Q-Learning", ShortcutEnvironment, alpha=0.5, epsilon=0.1, gamma=0.99)
run_single_and_render(ExpectedSARSAAgent, "SARSA", ShortcutEnvironment, alpha=0.5, epsilon=0.1, gamma=0.99)
run_single_and_render(ExpectedSARSAAgent, "Expected SARSA", ShortcutEnvironment, alpha=0.5, epsilon=0.2, gamma=0.99)

# Run all algorithms in windy environment
run_single_and_render(QLearningAgent, "Q-Learning (Windy)", WindyShortcutEnvironment, alpha=0.5, epsilon=0.1, gamma=0.95)
run_single_and_render(QLearningAgent, "SARSA (Windy)", WindyShortcutEnvironment, alpha=0.5, epsilon=0.1, gamma=0.95)



# # #----------------------------------------------------------------------
# # 1
# # Generate greedy path for Q-learning
# env = ShortcutEnvironment()
# agent = QLearningAgent(env.action_size(), env.state_size(), epsilon=0.2, alpha=0.5, gamma=0.99)
# returns = agent.train(env, 10000)
# grid = capture_render_greedy(env, agent.Q)
# plot_greedy_path(grid, "Q-Learning Greedy Path (Deterministic Environment)", "Q_learning_greedy_path.png")


# # Q-Learning, Testing different alpha values-------------------------
# alphas = [0.01, 0.1, 0.5, 0.9]
# n_rep = 100
# n_episodes = 1000
# alpha_curves = []
# for alpha in alphas:
#     all_returns = run_repetitions(ShortcutEnvironment, QLearningAgent, n_rep, n_episodes, alpha=alpha)
#     avg_returns = np.mean(all_returns, axis=0)
#     alpha_curves.append(avg_returns)

# plot_learning_curves(
#     alpha_curves,
#     'Q-Learning with Different Alpha Values',
#     labels=[f'alpha={a}' for a in alphas],
#     smooth_window=10
# )



# # #----------------------------------------------------------------------
# # 2
# # SARSA
# env = ShortcutEnvironment()
# sarsa_agent = SARSAAgent(env.action_size(), env.state_size(), epsilon=0.2, alpha=0.3, gamma=0.99)
# sarsa_returns = sarsa_agent.train(env, 10000)
# grid = capture_render_greedy(env, sarsa_agent.Q)
# plot_greedy_path(grid, "SARSA Greedy Path (Deterministic Environment)", "SARSA_greedy_path.png")


# # Structured SARSA Alpha Experiment
# # Test SARSA with different alpha values
# alphas = [0.01, 0.1, 0.5, 0.9]
# sarsa_alpha_curves = []

# for alpha in alphas:
#     returns = run_repetitions(ShortcutEnvironment, SARSAAgent, n_rep=100, n_episodes=1000, alpha=alpha)
#     avg_returns = np.mean(returns, axis=0)
#     sarsa_alpha_curves.append(avg_returns)

# # Plot SARSA alpha comparison
# plot_learning_curves(
#     sarsa_alpha_curves,
#     'SARSA with Different Alpha Values',
#     labels=[f'alpha={a}' for a in alphas],
#     smooth_window=10
# )


# # #-----------------------------------------------------------------------------------------------------------
# # 3
# # Windy Q-Learning
# windy_env = WindyShortcutEnvironment()
# q_agent = QLearningAgent(windy_env.action_size(), windy_env.state_size(), epsilon=0.3, alpha=0.3, gamma=0.95)
# q_returns = q_agent.train(windy_env, 10000)
# grid = capture_render_greedy(windy_env, q_agent.Q)
# plot_greedy_path(grid, "Q-Learning Greedy Path (Windy Environment)", "Q_learning_stochastic_path.png")

# # Windy SARSA
# windy_env = WindyShortcutEnvironment()
# sarsa_agent = SARSAAgent(windy_env.action_size(), windy_env.state_size(), epsilon=0.3, alpha=0.2, gamma=0.95)
# sarsa_returns = sarsa_agent.train(windy_env, 10000)
# grid = capture_render_greedy(windy_env, sarsa_agent.Q)
# plot_greedy_path(grid, "SARSA Greedy Path (Windy Environment)", "SARSA_stochastic_path.png")



# # # ---------------------------------------------------------------------------------------
# # 4
# # Expected SARSA
# env = ShortcutEnvironment()
# expected_sarsa_agent = ExpectedSARSAAgent(env.action_size(), env.state_size(), epsilon=0.2, alpha=0.4, gamma=0.99)
# expected_returns = expected_sarsa_agent.train(env, 10000)
# grid = capture_render_greedy(env, expected_sarsa_agent.Q)
# plot_greedy_path(grid, "Expected SARSA Greedy Path (Deterministic Environment)", "Expected_SARSA_greedy_path.png")

# # Windy Expected SARSA
# windy_env = WindyShortcutEnvironment()
# expected_agent = ExpectedSARSAAgent(windy_env.action_size(), windy_env.state_size(), epsilon=0.3, alpha=0.3, gamma=0.95)
# expected_returns = expected_agent.train(windy_env, 10000)
# grid = capture_render_greedy(windy_env, expected_agent.Q)
# plot_greedy_path(grid, "Expected SARSA Greedy Path (Windy Environment)", "Expected_SARSA_stochastic_path.png")


# # Test Expected SARSA with different alpha values
# alphas = [0.01, 0.1, 0.5, 0.9]
# expected_alpha_curves = []

# for alpha in alphas:
#     returns = run_repetitions(ShortcutEnvironment, ExpectedSARSAAgent, n_rep=100, n_episodes=1000, alpha=alpha)
#     avg_returns = np.mean(returns, axis=0)
#     expected_alpha_curves.append(avg_returns)

# # Plot Expected SARSA alpha comparison
# plot_learning_curves(
#     expected_alpha_curves,
#     'Expected SARSA with Different Alpha Values',
#     labels=[f'alpha={a}' for a in alphas],
#     smooth_window=10
# )



# # ----------------------------------------------------------------------
# # 5
# # Single n-step SARSA experiment with 10,000 episodes
# env = ShortcutEnvironment()
# n_step_agent = nStepSARSAAgent(
#     n_actions=env.action_size(),
#     n_states=env.state_size(),
#     n=5,
#     epsilon=0.1,
#     alpha=0.1
# )
# n_step_returns = n_step_agent.train(env, 10000)
# env.render_greedy(n_step_agent.Q)

# # Multiple repetitions for n-step SARSA
# n_step_returns = run_repetitions(
#     ShortcutEnvironment,
#     lambda: nStepSARSAAgent(n_actions=4, n_states=144, n=5, epsilon=0.1, alpha=0.1),
#     n_rep=100,
#     n_episodes=1000
# )
# n_step_avg = np.mean(n_step_returns, axis=0)

# # Plot comparison with other algorithms
# plot_learning_curves(
#     [q_avg, sarsa_avg, expected_avg, n_step_avg],
#     'Comparison of RL Algorithms',
#     labels=['Q-Learning', 'SARSA', 'Expected SARSA', 'n-Step SARSA (n=5)'],
#     smooth_window=10
# )


# # Test n-step SARSA with different alpha values
# n_values = [1, 2, 5, 10, 25]
# alpha = 0.1
# n_step_curves = []
# env = ShortcutEnvironment()

# for n in n_values:
#     returns = run_repetitions(
#         ShortcutEnvironment,
#         nStepSARSAAgent,
#         n_rep=100,
#         n_episodes=1000,
#         alpha=alpha,
#         n=n    # adding this line to make the n-steps work
#     )
#     avg_returns = np.mean(returns, axis=0)
#     n_step_curves.append(avg_returns)

# # Plot n-step comparison
# plot_learning_curves(
#     n_step_curves,
#     'n-Step SARSA with Different n Values',
#     labels=[f'n={n}' for n in n_values],
#     smooth_window=10
# )



# # # ----------------------------------------------------------------------------------------
# # # 6
# # # Comparison
# def generate_comparison_learning_curves():

#     q_returns = run_repetitions(
#         ShortcutEnvironment, 
#         QLearningAgent, 
#         n_rep=100, 
#         n_episodes=1000, 
#         alpha=0.9
#     )
#     q_avg = np.mean(q_returns, axis=0)

#     sarsa_returns = run_repetitions(
#         ShortcutEnvironment, 
#         SARSAAgent, 
#         n_rep=100, 
#         n_episodes=1000, 
#         alpha=0.5
#     )
#     sarsa_avg = np.mean(sarsa_returns, axis=0)

#     expected_returns = run_repetitions(
#         ShortcutEnvironment, 
#         ExpectedSARSAAgent, 
#         n_rep=100, 
#         n_episodes=1000, 
#         alpha=0.9
#     )
#     expected_avg = np.mean(expected_returns, axis=0)

#     n_step_returns = run_repetitions(
#         ShortcutEnvironment, 
#         nStepSARSAAgent, 
#         n_rep=100, 
#         n_episodes=1000, 
#         alpha=0.1, 
#         n=2
#     )
#     n_step_avg = np.mean(n_step_returns, axis=0)

#     plot_learning_curves(
#         [q_avg, sarsa_avg, expected_avg, n_step_avg],
#         title='Comparison of Best Performing Models',
#         labels=[
#             f'Q-Learning (α=0.9)', 
#             f'SARSA (α=0.5)', 
#             f'Expected SARSA (α=0.9)', 
#             f'n-step SARSA (n=2, α=0.1)'
#         ],
#         smooth_window=10
#     )
#     plt.close()

# generate_comparison_learning_curves()