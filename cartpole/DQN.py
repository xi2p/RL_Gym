import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym

# -------------------- 超参数 --------------------
ENV_NAME = "CartPole-v1"
MEMORY_SIZE = 10000        # 经验回放容量
BATCH_SIZE = 32            # 每次训练的样本数
GAMMA = 0.99               # 折扣因子
EPS_START = 1.0            # ε 初始值
EPS_END = 0.01             # ε 最小值
EPS_DECAY = 500            # ε 衰减步数（线性衰减）
TARGET_UPDATE = 100        # 目标网络更新频率（步数）
LEARNING_RATE = 1e-3
NUM_EPISODES = 10000         # 训练回合数
MAX_STEPS = 500            # 每回合最大步数
HIDDEN_DIM = 256           # 隐藏层神经元数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Q 网络（全连接）--------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# -------------------- 经验回放缓冲区 --------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32),
                np.array(next_state), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

# -------------------- 智能体 --------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不训练
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps_done = 0

    # ε-贪心策略
    def select_action(self, state, eps):
        if np.random.random() < eps:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    # 存储经验
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    # 从回放中采样并更新网络
    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        # 当前 Q 值
        current_q = self.policy_net(states).gather(1, actions)

        # 目标 Q 值：使用目标网络计算下一状态的最大 Q 值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * GAMMA * next_q

        # 损失函数：MSE
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络（硬更新）
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# -------------------- 训练主循环 --------------------
def train():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    episode_rewards = []
    moving_avg = []

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        # 当前回合的 ε 值（线性衰减）
        eps = max(EPS_END, EPS_START - episode / EPS_DECAY)

        for step in range(MAX_STEPS):
            action = agent.select_action(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

        # 移动平均（最后10回合）
        if len(episode_rewards) >= 10:
            avg = np.mean(episode_rewards[-10:])
            moving_avg.append(avg)
        else:
            moving_avg.append(total_reward)

        if (episode+1) % 50 == 0:
            print(f"Episode {episode+1}, Total Reward: {total_reward:.1f}, Eps: {eps:.3f}")

    env.close()

    # 绘制奖励曲线
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(moving_avg, label='10-episode Moving Avg', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN on CartPole')
    plt.legend()
    plt.grid(True)
    plt.savefig('dqn_cartpole.png')
    plt.show()

if __name__ == "__main__":
    train()