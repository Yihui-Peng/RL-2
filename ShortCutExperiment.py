# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.

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
            **agent_kwargs  # adding this line to make the n-steps work
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
    

def windy_experiments():
    # Q-Learning Experiment
    windy_env = WindyShortcutEnvironment()
    q_agent = QLearningAgent(windy_env.action_size(), windy_env.state_size(), epsilon=0.1, alpha=0.1)
    q_returns = q_agent.train(windy_env, 10000)
    print("Q-Learning Greedy Policy (Windy Environment):")
    windy_env.render_greedy(q_agent.Q)

    # SARSA Experiment
    windy_env_sarsa = WindyShortcutEnvironment()
    sarsa_agent = SARSAAgent(windy_env_sarsa.action_size(), windy_env_sarsa.state_size(), epsilon=0.1, alpha=0.1)
    sarsa_returns = sarsa_agent.train(windy_env_sarsa, 10000)
    print("SARSA Greedy Policy (Windy Environment):")
    windy_env_sarsa.render_greedy(sarsa_agent.Q)


# # Run experiments
# windy_experiments()


# #----------------------------------------------------------------------
# # Single experiment with 10000 episodes
# env = ShortcutEnvironment()
# agent = QLearningAgent(env.action_size(), env.state_size(), epsilon=0.1, alpha=0.1)
# returns = agent.train(env, 10000)
# env.render_greedy(agent.Q)

# # Multiple repetitions (100 reps, 1000 episodes)
# n_rep = 100
# n_episodes = 1000
# all_returns = run_repetitions(ShortcutEnvironment, QLearningAgent, n_rep, n_episodes)
# avg_returns = np.mean(all_returns, axis=0)
# plot_learning_curves([avg_returns], 'Average Learning Curve (Q-Learning)', smooth_window=10)


# # Testing different alpha values
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



# #----------------------------------------------------------------------
# # Single SARSA experiment with 10,000 episodes
# env = ShortcutEnvironment()
# sarsa_agent = SARSAAgent(env.action_size(), env.state_size(), epsilon=0.1, alpha=0.1)
# sarsa_returns = sarsa_agent.train(env, 10000)
# env.render_greedy(sarsa_agent.Q)

# # Multiple repetitions for SARSA
# sarsa_returns = run_repetitions(ShortcutEnvironment, SARSAAgent, n_rep=100, n_episodes=1000)
# sarsa_avg = np.mean(sarsa_returns, axis=0)

# # Compare with Q-learning results from part 1b
# q_learning_returns = run_repetitions(ShortcutEnvironment, QLearningAgent, n_rep=100, n_episodes=1000)
# q_avg = np.mean(q_learning_returns, axis=0)

# # Plot comparison
# plot_learning_curves(
#     [q_avg, sarsa_avg],
#     'Q-Learning vs SARSA',
#     labels=['Q-Learning', 'SARSA'],
#     smooth_window=10
# )


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




# # ----------------------------------------------------------------------
# # Single Expected SARSA experiment with 10,000 episodes
# env = ShortcutEnvironment()
# expected_sarsa_agent = ExpectedSARSAAgent(env.action_size(), env.state_size(), epsilon=0.1, alpha=0.1)
# expected_returns = expected_sarsa_agent.train(env, 10000)
# env.render_greedy(expected_sarsa_agent.Q)

# # Multiple repetitions for Expected SARSA
# expected_returns = run_repetitions(ShortcutEnvironment, ExpectedSARSAAgent, n_rep=100, n_episodes=1000)
# expected_avg = np.mean(expected_returns, axis=0)

# # Plot comparison with Q-Learning and SARSA
# plot_learning_curves(
#     [q_avg, sarsa_avg, expected_avg],
#     'Q-Learning vs SARSA vs Expected SARSA',
#     labels=['Q-Learning', 'SARSA', 'Expected SARSA'],
#     smooth_window=10
# )


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


# Test n-step SARSA with different alpha values
n_values = [1, 2, 5, 10, 25]
alpha = 0.1
n_step_curves = []
env = ShortcutEnvironment()

for n in n_values:
    returns = run_repetitions(
        ShortcutEnvironment,
        nStepSARSAAgent,
        n_rep=100,
        n_episodes=1000,
        alpha=alpha,
        n=n    # adding this line to make the n-steps work
    )
    avg_returns = np.mean(returns, axis=0)
    n_step_curves.append(avg_returns)

# Plot n-step comparison
plot_learning_curves(
    n_step_curves,
    'n-Step SARSA with Different n Values',
    labels=[f'n={n}' for n in n_values],
    smooth_window=10
)