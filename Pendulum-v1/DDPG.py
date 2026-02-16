"""
Deep Deterministic Policy Gradient (DDPG) 实现
环境: Pendulum-v1 (连续动作)
核心组件: Actor, Critic, Replay Buffer, Target Networks, Soft Update
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# ==================== 超参数 ====================
ENV_NAME = "Pendulum-v1"
MAX_EPISODES = 200
MAX_STEPS = 1000
GAMMA = 0.99
TAU = 0.005           # 软更新系数
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EXPLORATION_NOISE = 0.1  # 探索噪声标准差
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Actor网络 ====================
class Actor(nn.Module):
    """输入状态，输出确定性动作"""
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        self.max_action = max_action  # 缩放到实际动作范围

    def forward(self, state):
        return self.max_action * self.net(state)

# ==================== Critic网络 ====================
class Critic(nn.Module):
    """输入状态+动作，输出Q值"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        # 确保action维度正确（兼容batch）
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        sa = torch.cat([state, action], dim=1)
        return self.net(sa)

# ==================== 经验回放 ====================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(np.array(state)).to(DEVICE),
                torch.FloatTensor(np.array(action)).to(DEVICE),
                torch.FloatTensor(np.array(reward)).unsqueeze(1).to(DEVICE),
                torch.FloatTensor(np.array(next_state)).to(DEVICE),
                torch.FloatTensor(np.array(done)).unsqueeze(1).to(DEVICE))

    def __len__(self):
        return len(self.buffer)

# ==================== DDPG智能体 ====================
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.target_actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.target_critic = Critic(state_dim, action_dim).to(DEVICE)

        # 初始化目标网络 = 主网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        self.max_action = max_action

    def select_action(self, state, noise=True):
        """选择确定性动作 + 可选探索噪声"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        action = self.actor(state_tensor).detach().cpu().numpy().flatten()
        if noise:
            # 添加高斯噪声进行探索
            noise = np.random.normal(0, EXPLORATION_NOISE * self.max_action, size=action.shape)
            action = action + noise
            # 截断到合法范围
            action = np.clip(action, -self.max_action, self.max_action)
        return action

    def update(self):
        """从经验池采样一个batch，更新一次网络"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        # ----- 1. 更新 Critic（Q学习目标）-----
        with torch.no_grad():
            # 目标策略网络输出下一动作
            next_actions = self.target_actor(next_states)
            # 目标Q网络输出目标Q值
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + GAMMA * (1 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----- 2. 更新 Actor（确定性策略梯度）-----
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----- 3. 软更新目标网络 -----
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# ==================== 训练主循环 ====================
def train():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])  # Pendulum: 2.0

    agent = DDPGAgent(state_dim, action_dim, max_action)

    episode_rewards = []

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state, noise=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            # 每步都进行训练（DDPG是off-policy，可以每步更新）
            agent.update()

            if done:
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1}, Avg Reward (last 20): {avg_reward:.2f}")

    env.close()

    # 绘制奖励曲线
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'DDPG on {ENV_NAME}')
    plt.grid()
    plt.savefig('ddpg_pendulum.png')
    plt.show()

if __name__ == "__main__":
    train()