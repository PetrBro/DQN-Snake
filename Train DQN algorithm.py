import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
from collections import deque
import pygame
from SnakeEnv import SnakeEnv
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 2.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([x[0] for x in minibatch]).to(self.device)
        actions = torch.LongTensor([x[1] for x in minibatch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in minibatch]).to(self.device)
        next_states = torch.FloatTensor([x[3] for x in minibatch]).to(self.device)
        dones = torch.BoolTensor([x[4] for x in minibatch]).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones.float()) * self.gamma * next_q

        loss = self.loss_fn(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.decay_epsilon()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Обучение с визуализацией
if __name__ == "__main__":
    env = SnakeEnv(grid_size=20, render=True)
    state_size = len(env.reset())
    action_size = 3

    episodes_reward = []

    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32
    update_target_freq = 50

    try:
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                agent.replay(batch_size)

                # Обработка событий Pygame
                if env.render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            exit()

            if e % update_target_freq == 0:
                agent.update_target_model()

            print(
                f"Episode: {e + 1}/{episodes}, Score: {env.reward}, Epsilon: {agent.epsilon:.2f}")

            episodes_reward.append(env.reward)
    finally:
        env.close()
        torch.save(agent.policy_net.state_dict(), r'C:\py projects\pythonProject\Previous projects\snake_dqn.pth')

        plt.plot(np.arange(1, episodes), episodes_reward)