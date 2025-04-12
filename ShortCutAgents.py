import numpy as np
from collections import deque

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))


    def select_action(self, state):
        # TO DO: Implement policy
        # action = None
        # return action
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.Q[state, :]
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            if 3 in max_actions:
                return 3
            elif 1 in max_actions:
                return 1
            return np.random.choice(max_actions)
        

    def update(self, state, action, reward, next_state, done): # Augment arguments if necessary
        # TO DO: Implement Q-learning update
        current_q = self.Q[state, action]
        if done:
            target = reward
        else:
            max_next_q = np.max(self.Q[next_state, :])
            target = reward + self.gamma * max_next_q
        self.Q[state, action] += self.alpha * (target - current_q)

    
    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        # episode_returns = []
        # return episode_returns
        episode_returns = []
        initial_epsilon = self.epsilon
        for episode in range(n_episodes):
            self.epsilon = max(0.01, initial_epsilon * (0.9 ** (episode / (n_episodes // 10))))
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                self.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
            episode_returns.append(total_reward)
        return episode_returns


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))


    def select_action(self, state):
        # TO DO: Implement policy
        # action = None
        # return action
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.Q[state, :]
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            if 3 in max_actions:
                return 3
            elif 1 in max_actions:
                return 1
            return np.random.choice(max_actions)
        
        
    def update(self, state, action, reward, next_state, next_action, done): # Augment arguments if necessary
        # TO DO: Implement SARSA update
        target = reward if done else reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        # episode_returns = []
        # return episode_returns
        episode_returns = []
        initial_epsilon = self.epsilon
        for episode in range(n_episodes):
            self.epsilon = max(0.01, initial_epsilon * (0.9 ** (episode / (n_episodes // 10))))
            state = env.reset()
            action = self.select_action(state)
            done = False
            total_reward = 0
            while not done:
                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                next_action = self.select_action(next_state) if not done else None
                self.update(state, action, reward, next_state, next_action, done)
                state, action = next_state, next_action
                total_reward += reward
            episode_returns.append(total_reward)
        return episode_returns

class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy
        # action = None
        # return action
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.Q[state, :]
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            if 3 in max_actions:  # 向右优先
                return 3
            elif 1 in max_actions:  # 向下次优先
                return 1
            return np.random.choice(max_actions)
        
    def update(self, state, action, reward, next_state, done): # Augment arguments if necessary
        # TO DO: Implement Expected SARSA update
        current_q = self.Q[state, action]
        if done:
            target = reward
        else:
            q_next = self.Q[next_state, :]
            max_action = np.argmax(q_next)
            prob_non_greedy = self.epsilon / self.n_actions
            prob_greedy = (1 - self.epsilon) + prob_non_greedy
            expected_q = np.sum(q_next * prob_non_greedy)
            expected_q += q_next[max_action] * (prob_greedy - prob_non_greedy)
            target = reward + self.gamma * expected_q
        self.Q[state, action] += self.alpha * (target - current_q)

    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        # episode_returns = []
        # return episode_returns 
        episode_returns = []
        initial_epsilon = self.epsilon
        for episode in range(n_episodes):
            self.epsilon = max(0.01, initial_epsilon * (0.9 ** (episode / (n_episodes // 10))))
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                self.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
            episode_returns.append(total_reward)
        return episode_returns 


class nStepSARSAAgent(object):

    def __init__(self, n_actions, n_states, n, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy
        # action = None
        # return action
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.Q[state, :]
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            if 3 in max_actions:
                return 3
            elif 1 in max_actions:
                return 1
            return np.random.choice(max_actions)
        
    def update(self, states, actions, rewards, done): # Augment arguments if necessary
        # TO DO: Implement n-step SARSA update
        pass
    
    def train(self, env, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        initial_epsilon = self.epsilon
        episode_returns = []
        for episode in range(n_episodes):
            self.epsilon = max(0.01, initial_epsilon * (0.9 ** (episode / (n_episodes // 10))))
            state = env.reset()
            done = False
            total_reward = 0
            buffer = deque(maxlen=self.n + 1)
            while not done:
                action = self.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                buffer.append((state, action, reward, next_state, done))
                if len(buffer) == self.n + 1:
                    s_tau, a_tau, _, _, _ = buffer[0]
                    G = 0
                    for k in range(self.n):
                        G += (self.gamma ** k) * buffer[k + 1][2]
                    s_next, a_next = buffer[self.n][0], buffer[self.n][1]
                    G += (self.gamma ** self.n) * self.Q[s_next, a_next]
                    self.Q[s_tau, a_tau] += self.alpha * (G - self.Q[s_tau, a_tau])
                total_reward += reward
                state = next_state
            for i in range(len(buffer)):
                s_tau, a_tau, _, _, _ = buffer[i]
                G = 0
                for k in range(i, min(i + self.n, len(buffer))):
                    G += (self.gamma ** (k - i)) * buffer[k][2]
                if i + self.n < len(buffer):
                    s_next, a_next = buffer[i + self.n][0], buffer[i + self.n][1]
                    G += (self.gamma ** self.n) * self.Q[s_next, a_next]
                self.Q[s_tau, a_tau] += self.alpha * (G - self.Q[s_tau, a_tau])
            episode_returns.append(total_reward)
        return episode_returns
    
    
    