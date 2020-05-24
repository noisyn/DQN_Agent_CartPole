'''Dueling Deep Q-Network (DQN) agent for training/playing >>Classical control<<
environments of OpenAI Gym (https://gym.openai.com/). 
'''

# The MIT License

# Copyright (c) 2020 Taner Esat <taner@esat.xyz> (http://esat.xyz & http://noisyn.de/)

# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import datetime
from collections import deque, namedtuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Namedtuple for storing transitions/experiences
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceBuffer():
    """Experience/Replay buffer for storing and sampling transitions."""
    def __init__(self, capacity):
        """Initializes a deque object for storing experiences.

        Args:
            capacity (int): Maximum capacity of the buffer.
        """
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
    
    def append(self, transition):
        """Appends new transition to the replay buffer.

        Args:
            transition (namedtuple('Transition')): Experience to be stored.
        """
        self.memory.append(transition)

    def sample(self, batch_size):
        """Samples transitions/experiences from the replay buffer.

        Args:
            batch_size (int): Size of the batch to be sampled.

        Returns:
            [list]: Randomly sampled transitions from the replay buffer.
        """
        indices = np.random.choice(self.capacity, size=batch_size, replace=False)
        batch = [self.memory[x] for x in indices]
        return batch

    def filled(self):
        """Indicates if the maximum capacity of the buffer is reached.

        Returns:
            [bool]: True if the maximum capacity is reached, else False.
        """
        if len(self.memory) == self.capacity:
            return True
        else:
            return False

class DQNAgent():
    """Dueling Deep Q-Network (DQN) agent for training/playing >>Classical control<<
    environments of OpenAI Gym. 
    """
    def __init__(self, env):
        """Initializes active and target networks.

        Args:
            env (gym.env): Environment to interact with.
        """
        self.env = env

        self.num_actions = env.action_space.n
        self.num_inputs = env.observation_space.shape[0]

        self.active_network = self._build_network(self.num_inputs, self.num_actions)
        self.target_network = self._build_network(self.num_inputs, self.num_actions)
        self._synchronize_networks()

    def _build_network(self, num_inputs, num_outputs):
        """Builds a dueling deep Q-Network architecture.

        Args:
            num_inputs (int): Number of observations.
            num_outputs (int): Number of actions.

        Returns:
            tf.keras.Model: Compiled model of the neural network.
        """

        # Build dueling Q-network architecture
        inputs = tf.keras.Input(shape=(self.num_inputs,))
        layer1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)(inputs)
        layer2_adv = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)(layer1)
        layer2_val = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)(layer1)
        advantages = tf.keras.layers.Dense(units=num_outputs)(layer2_adv)
        values = tf.keras.layers.Dense(units=1)(layer2_val)
        q_values = advantages + values - tf.math.reduce_mean(advantages, axis=1, keepdims=True)
        model = tf.keras.Model(inputs=inputs, outputs=q_values)

        adam_optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=adam_optimizer, loss='mse')

        return model

    def _update_network(self):
        """Updates the weights of the active network."""
        batch = self.experience.sample(self.batch_size)

        x = np.zeros(shape=(self.batch_size, self.num_inputs))
        y = np.zeros(shape=(self.batch_size, self.num_actions))

        for idx, trans in enumerate(batch):
            state, action, reward, next_state, done = trans

            # Bellman approximation
            if done:
                target = reward
            else:
                next_state_q = self.target_network.predict_on_batch(next_state)[0]
                target = reward + self.gamma * np.amax(next_state_q)
            
            current_state_q = self.active_network.predict_on_batch(state)[0]

            mask = np.ones(shape=(self.num_actions,))
            mask[action] = 0
            expected_q_values = current_state_q*mask + target*(1-mask) 

            x[idx] = state.copy()
            y[idx] = expected_q_values
        
        self.active_network.train_on_batch(x, y)

    def _synchronize_networks(self):
        """Copies the weights from the active network
         to the target network."""
        for i, layer in enumerate(self.active_network.layers):
            weights = layer.get_weights()
            self.target_network.layers[i].set_weights(weights)

    def _save_network(self, path):
        """Saves the weights of the active network.

        Args:
            path (str): Path where the weights will be saved.
        """
        self.active_network.save_weights(path)
        
    def load_network(self, path):
        """Loads the saved weights from the specified path
         to the active and target network. 

        Args:
            path (str): Path to the weights which will be loaded.
        """
        self.active_network.load_weights(path)
        self._synchronize_networks()

    def train(self, replay_size=2000, max_episodes=250, max_steps=300, batch_size=48, 
            sync_steps=100, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=0.995, 
            gamma=0.9):
        """Trains the DQN agent by interacting with the environment.

        Args:
            replay_size (int, optional): Size of the experience/replay buffer. 
                Defaults to 2000.
            max_episodes (int, optional): Maximum number of episodes. 
                Defaults to 250.
            max_steps (int, optional): Maximum number of actions/steps per episode. 
                Defaults to 300.
            batch_size (int, optional): Batch size sampled from the replay buffer. 
                Defaults to 48.
            sync_steps (int, optional): After how many actions/steps the active and 
                target network will be synchronized. Defaults to 100.
            epsilon_start (float, optional): Initial value of epsilon. Defaults to 1.0.
            epsilon_final (float, optional): Final value of epsilon. Defaults to 0.01.
            epsilon_decay (float, optional): Decay rate of epsilon. Defaults to 0.995.
            gamma (float, optional): Discount factor. Defaults to 0.9.
        """
        self.experience = ExperienceBuffer(replay_size)
        
        self.episodes = max_episodes
        self.steps = max_steps
        self.sync_steps = sync_steps

        self.training_episodes = []
        self.training_rewards = []
        
        self.batch_size = batch_size
        
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.gamma = gamma

        # Use Tensorboard to log total rewards 
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        total_steps = 0
        for episode in range(self.episodes):
            episode_reward = 0
            observation = self.env.reset()
            for _ in range(self.steps):
                observation, reward, done = self._training_step(observation)
                episode_reward += reward
                total_steps += 1
                
                # Start training/updating the networks only when the replay 
                # buffer is filled completely
                if self.experience.filled():
                    self._update_network()
                    self._update_epsilon()
                    if total_steps % self.sync_steps == 0:
                        self._synchronize_networks()

                # Log rewards
                if done:
                    self.training_episodes.append(episode)
                    self.training_rewards.append(episode_reward)
                    with train_summary_writer.as_default():
                        tf.summary.scalar('total_reward', episode_reward, step=episode)
                    break

            print('Episode {}, Total reward: {}'.format(episode, episode_reward))
        
        self._save_network('logs/' + current_time + '/trained_network')
        self._plot_performance(self.training_episodes, self.training_rewards, "Training performance")
      
    def _training_step(self, observation):
        """Executes a single training step (action) in the environment
        and saves the observations of the current and next state, the reward 
        and status of the environment (terminated or not) in the replay buffer.

        Args:
            observation (ndarray): Array representing the current 
                observations of the environment.

        Returns:
            ndarray, float, bool: Observation, reward and status 
                (terminated or not) of the environment after taking an action.
        """
        current_state = np.reshape(observation, [-1, self.num_inputs])
        
        # Exploration vs Exploitation
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            q_values = self.active_network.predict_on_batch(current_state)[0]
            action = np.argmax(q_values)

        observation, reward, done, _ = self.env.step(action)
        next_state = np.reshape(observation, [-1, self.num_inputs])

        # Store experience in replay buffer
        self.experience.append(Transition(current_state, action, reward, next_state, done))

        return observation, reward, done

    def _update_epsilon(self):
        """Decays and updates epsilon"""
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_final

    def _plot_performance(self, x, y, title='Performance'):
        """Plots the performance of the training or game play.

        Args:
            x (list): Episodes.
            y (list): Total rewards.
            title (str, optional): Title of the plot. Defaults to 'Performance'.
        """
        plt.figure(figsize=(9,4))
        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.show()

    def play(self, max_episodes=250, max_steps=300, rendering=False):
        """Plays/runs the environment using the current active network.

        Args:
            max_episodes (int, optional): Maximum number of episodes to be played. 
                Defaults to 250.
            max_steps (int, optional): Maximum number of actions/steps per episode. 
                Defaults to 300.
            rendering (bool, optional): If the environment should be rendered or not. 
                Defaults to False.
        """
        play_episodes = []
        play_rewards = []

        # Use Tensorboard to log total rewards 
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        play_log_dir = 'logs/' + current_time + '/play'
        play_summary_writer = tf.summary.create_file_writer(play_log_dir)

        for episode in range(max_episodes):
            observation = self.env.reset()
            episode_reward = 0
            for _ in range(max_steps):
                if rendering:
                    self.env.render()

                # Use active network to interact with environment
                current_state = np.reshape(observation, [-1, self.num_inputs])
                q_values = self.active_network.predict_on_batch(current_state)[0]
                action = np.argmax(q_values)
                
                observation, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # Log rewards
                if done:
                    play_episodes.append(episode)
                    play_rewards.append(episode_reward)
                    with play_summary_writer.as_default():
                        tf.summary.scalar('total_reward', episode_reward, step=episode)
                    break
        
        self._plot_performance(play_episodes, play_rewards, title='Play performance')
    

if __name__ == '__main__':
    # CartPole environment
    env = gym.make('CartPole-v0')
    # Example how to use DQN agent
    agent = DQNAgent(env)
    agent.train()
    # Possible to load pre-trained networks --> for playing and/or training
    # agent.load_network('logs/20200523-221335/trained_network')
    agent.play()
    env.close()